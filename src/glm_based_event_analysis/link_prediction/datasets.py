from transformers import AutoTokenizer
import numpy as np
import json
import networkx as nx
from typing import Optional
from abc import ABC, abstractmethod
from functools import partial
from datasets import Dataset, ClassLabel
from tqdm import tqdm
from glm_based_event_analysis.utils.sampling import sample_neg_edges
from glm_based_event_analysis.random_walks.neighbourhood_serialization import NeighbourhoodSerializer
from glm_based_event_analysis.random_walks.sampler import RandomWalkSampler
from glm_based_event_analysis.link_prediction.prompt_templates import FT_USER_PROMPT, FT_USER_PROMPT_WITH_NEIGHBOURS

# template alternativo ao json
EVENT_STR_TEMPLATE = """<event>
When: {when}
What: {what}
Who: {who}
Why: {why}
Where: {where}
How: {how}
ID: {id}
</event>
"""

def link_prediction_preprocessing(examples: dict, tokenizer: AutoTokenizer, add_generation_prompt: bool = False) -> dict:
    """
    Pre-processing function for link prediction datasets.
    It takes a batch of examples containing pairs of events (u, v) and their labels. 
    All components (u, v and label) must be dictionaries that can be converted to JSON strings.
    """

    json_strs = []


    for u, v, label in zip(examples["source"], examples["target"], examples["label"]):
        u_str = json.dumps(u, indent=2)
        v_str = json.dumps(v, indent=2)

        user_prompt = FT_USER_PROMPT.format(u_str, v_str)

        msgs = [
            {'role': 'user','content': user_prompt},
        ]

        # caso esta criando rótulos para treino, é preciso incluir o rótulo
        # caso seja para teste, o modelo deve gerar o rótulo, 
        if not add_generation_prompt:
            label_str = json.dumps(label, indent=2) # TODO: provisório. no futuro essa expl deve vir nos rotulos tb
            msgs.append({'role': 'assistant','content': label_str})

        json_strs.append(
            tokenizer.apply_chat_template(msgs, 
                        tokenize=False, 
                        add_generation_prompt=add_generation_prompt
            ).removeprefix('<bos>') 
        )


    return {"text": json_strs}

def link_prediction_preprocessing_with_neighbours(examples: dict, tokenizer: AutoTokenizer, add_generation_prompt: bool = False) -> dict:
    """
    Pre-processing function for link prediction datasets.
    It takes a batch of examples containing pairs of events (u, v) and their labels. 
    All components (u, v and label) must be dictionaries that can be converted to JSON strings.
    """

    json_strs = []


    for u, v, neighbours, label in zip(examples["source"], examples["target"], examples["source_neighbours"], examples["label"]):

        u_str = json.dumps(u, indent=2)
        v_str = json.dumps(v, indent=2)
        neighbours_str = json.dumps(neighbours, indent=2)

        user_prompt = FT_USER_PROMPT_WITH_NEIGHBOURS.format(u_str, v_str, neighbours_str)

        msgs = [
            {'role': 'user','content': user_prompt},
        ]

        # caso esta criando rótulos para treino, é preciso incluir o rótulo
        # caso seja para teste, o modelo deve gerar o rótulo, 
        if not add_generation_prompt:
            label_str = json.dumps(label, indent=2) # TODO: provisório. no futuro essa expl deve vir nos rotulos tb
            msgs.append({'role': 'assistant','content': label_str})

        json_strs.append(
            tokenizer.apply_chat_template(msgs, 
                        tokenize=False, 
                        add_generation_prompt=add_generation_prompt
            ).removeprefix('<bos>') 
        )


    return {"text": json_strs}

def causal_lm_preprocessing(examples: dict, eos_token: str) -> dict:
    """
    Adds EOS tokens as a simple pre-processing step for the causal LM dataset.
    """

    return {"text": [example + eos_token for example in examples["text"]]} 

class BaseDatasetBuilder(ABC):

    def __init__(self, tokenizer: AutoTokenizer,
                       G_ref: nx.Graph):

        self._tokenizer: AutoTokenizer = tokenizer
        self._G: nx.Graph = G_ref # usado apenas para consulta de metadados. não deve ser modificado

        # debug
        self.walk_lens = []


    def _get_node_metadata(self, nodeid: str) -> dict:

        """Obtains the metadadata of a node, from its attributes in the reference graph."""

        node_data = self._G.nodes[nodeid]
        node_data['event_id'] = nodeid
        curr_date = node_data.get('when', None)

        if isinstance(curr_date, (str)):
            node_data['when'] = curr_date
        else:
            node_data['when'] = curr_date.strftime("%Y-%m-%d-%H:%M:%S")

        return node_data
    
    def _extract_node_information_from_edges(self, edges: list[tuple]) -> tuple[list[str], list[str]]:

        source_data = []
        target_data = []


        for edge in edges:
            u, v = edge
            u_data = self._get_node_metadata(u)
            v_data = self._get_node_metadata(v)

            source_data.append(u_data)
            target_data.append(v_data)

        return source_data, target_data
    
    @abstractmethod
    def build_train_ds(self, train_edges: list[tuple], train_labels: list[bool]) -> Dataset:
        """
        Builds the training dataset for link prediction. 
        Receives a list of edges and their corresponding labels, and returns a HuggingFace Dataset ready for training.
        """
        
        raise NotImplementedError

    @abstractmethod
    def build_test_ds(self, test_edges: list[tuple], test_labels: list[bool]) -> Dataset:
        """
        Builds the test dataset for link prediction. 
        Receives a list of edges and their corresponding labels, and returns a HuggingFace Dataset ready for testing.
        """

        raise NotImplementedError
    

class LinkPredictionDatasetBuilder(BaseDatasetBuilder):

    def __init__(self, tokenizer: AutoTokenizer,
                       G_ref: nx.Graph):

        super().__init__(tokenizer, G_ref)

        # pre-procs 
        self._train_preprocessing_fn: callable = partial(link_prediction_preprocessing, tokenizer=self._tokenizer, add_generation_prompt=False)
        self._test_preprocessing_fn: callable = partial(link_prediction_preprocessing, tokenizer=self._tokenizer, add_generation_prompt=True)

    
    def build_train_ds(self, train_edges: list[tuple], train_labels: list[bool]) -> Dataset:
        """
            Builds the training dataset for link prediction.
        Params:
            train_edges: list of tuples (u, v) representing edges in the graph. u and v are node IDs.
            train_labels: list of booleans representing the label for each edge (True if edge exists, False otherwise).
        Returns:
            A HuggingFace Dataset object with the training data, ready for fine-tuning.
        """

        source_data, target_data = self._extract_node_information_from_edges(train_edges)
        labels =[{'label': label, "explanation": "placeholder"} for label in train_labels] # TODO: deve ser modificado no futuro com as explicações.

        # criando o dataset
        train_ds = Dataset.from_dict({
            "source": source_data,
            "target": target_data,
            "label": labels
        }) 

        # aplicando o pre-processamento (criação do prompt)
        train_ds = train_ds.map(self._train_preprocessing_fn, batched=True, remove_columns=train_ds.column_names) # os campos de source, target e label sao removidos apos a criacao do prompt, pois o modelo só precisa do campo 'text' para treinar

        # adicionando uma coluna com apenas os labels
        train_labels = list(map(int, train_labels)) 
        train_ds = train_ds.add_column("label", train_labels, feature=ClassLabel(num_classes=2)) # TODO colocar apenas as as strings dos rotulos aqui, sem explicações

        # embaralhando exemplos 
        train_ds = train_ds.shuffle(seed=2026)

        # note que, no ds, os exemplos correspondem a strings no campo 'text', que já estão prontas para serem tokenizadas e passadas para o modelo.
        return train_ds

    def build_test_ds(self, test_edges: list[tuple], test_labels: list[bool]) -> Dataset:
        """
        Builds the test dataset for link prediction.
        Params:
            test_edges: list of tuples (u, v) representing edges in the graph. u and v are node IDs.
            test_labels: list of booleans representing the label for each edge (True if edge exists, False otherwise).
        Returns:
            A HuggingFace Dataset object with the test data, ready for evaluation.
        """

        source_data, target_data = self._extract_node_information_from_edges(test_edges)

        # criando o dataset
        test_ds = Dataset.from_dict({
            "source": source_data,
            "target": target_data,  
            "label": test_labels # não é necessário criar JSONs para os rotulos de teste.
        })

        # aplicando o pre-processamento (criação do prompt)
        test_ds = test_ds.map(self._test_preprocessing_fn, batched=True, remove_columns=["source", "target"])
        
        return test_ds
    

class LinkPredictionDatasetBuilderWithNeighbours(BaseDatasetBuilder):

    def __init__(self, tokenizer: AutoTokenizer,
                       G_ref: nx.Graph,
                       neighbourhood_serializer: NeighbourhoodSerializer):

        super().__init__(tokenizer, G_ref)

        self._neighbourhood_serializer = neighbourhood_serializer

        # pre-procs 
        self._train_preprocessing_fn: callable = partial(link_prediction_preprocessing_with_neighbours, tokenizer=self._tokenizer, add_generation_prompt=False)
        self._test_preprocessing_fn: callable = partial(link_prediction_preprocessing_with_neighbours, tokenizer=self._tokenizer, add_generation_prompt=True)
    
    def build_train_ds(self, train_edges: list[tuple], train_labels: list[bool]) -> Dataset:
        """
            Builds the training dataset for link prediction, including neighborhood information.
        Params:
            train_edges: list of tuples (u, v) representing edges in the graph. u and v are node IDs.
            train_labels: list of booleans representing the label for each edge (True if edge exists, False otherwise).
        Returns:
            A HuggingFace Dataset object with the training data, ready for fine-tuning.
        """ 

        source_data, target_data = self._extract_node_information_from_edges(train_edges)
         # obtendo a vizinhança de u para cada aresta
        source_neighbours = [self._neighbourhood_serializer.get_neighbourhood_representation(u, return_json_str=False) for u, _ in train_edges]
    
        labels =[{'label': label, "explanation": "placeholder"} for label in train_labels] # TODO: deve ser modificado no futuro com as explicações.

        # criando o dataset
        train_ds = Dataset.from_dict({
            "source": source_data,
            "target": target_data,
            "source_neighbours": source_neighbours,
            "label": labels
        }) 

        # aplicando o pre-processamento (criação do prompt)
        train_ds = train_ds.map(self._train_preprocessing_fn, batched=True, remove_columns=train_ds.column_names) # os campos de source, target e label sao removidos apos a criacao do prompt, pois o modelo só precisa do campo 'text' para treinar

        # adicionando uma coluna com apenas os labels
        train_labels = list(map(int, train_labels)) 
        train_ds = train_ds.add_column("label", train_labels, feature=ClassLabel(num_classes=2)) # TODO colocar apenas as as strings dos rotulos aqui, sem explicações

        # embaralhando exemplos 
        train_ds = train_ds.shuffle(seed=2026)

        # note que, no ds, os exemplos correspondem a strings no campo 'text', que já estão prontas para serem tokenizadas e passadas para o modelo.
        return train_ds
    
    def build_test_ds(self, test_edges: list[tuple], test_labels: list[bool]) -> Dataset:
        """
        Builds the test dataset for link prediction, including neighborhood information.
        Params:
            test_edges: list of tuples (u, v) representing edges in the graph. u and v are node IDs.
            test_labels: list of booleans representing the label for each edge (True if edge exists, False otherwise).  
        Returns:
            A HuggingFace Dataset object with the test data, ready for evaluation.
        """

        source_data, target_data = self._extract_node_information_from_edges(test_edges)
        # obtendo a vizinhança de u para cada aresta
        source_neighbours = [self._neighbourhood_serializer.get_neighbourhood_representation(u, return_json_str=False) for u, _ in test_edges]
    

        # criando o dataset
        test_ds = Dataset.from_dict({
            "source": source_data,
            "target": target_data,
            "source_neighbours": source_neighbours,
            "label": test_labels # não é necessário criar JSONs para os rotulos de teste.
        })

        # aplicando o pre-processamento (criação do prompt)
        test_ds = test_ds.map(self._test_preprocessing_fn, batched=True, remove_columns=["source", "target", "source_neighbours"]) # os campos de source, target e label sao removidos apos a criacao do prompt, pois o modelo só precisa do campo 'text' para treinar

        return test_ds

class CausalLMDatasetBuilder(BaseDatasetBuilder):

    def __init__(self, tokenizer: AutoTokenizer,
                       G_ref: nx.Graph,
                       bias: str = "none",
                       num_walks_per_node: int = 10,
                       walk_length: int = 5,
                       use_json_serialization: bool = False):

        super().__init__(tokenizer, G_ref)

        # parametros de random walks
        self._bias: str = bias
        self._num_walks_per_node: int = num_walks_per_node
        self._walk_length: int = walk_length

        # usar ou nao serialização em JSON para as caminhadas. 
        self._use_json_serialization: bool = use_json_serialization

        # sampler
        self._sampler = RandomWalkSampler(
            bias=self._bias,
            num_walks_per_node=self._num_walks_per_node,
            walk_length=self._walk_length
        )

        self._preprocessing_fn: callable = partial(causal_lm_preprocessing, eos_token=tokenizer.eos_token)

    def build_train_ds(self, train_edges: list[tuple], train_labels: list[bool]) -> Dataset:
        return

    def build_test_ds(self, test_edges: list[tuple], test_labels: list[bool]) -> Dataset:
        return

    def build_train_eval_ds(self, eval_pc: float = 0.1) -> Dataset:
        """
        Builds a dataset for training or evaluating a causal language model on the graph data, using random walks for data augmentation.
        Returns:
            A HuggingFace Dataset object with the training/evaluation data, ready for fine-tuning or evaluation.
        """

        # obtendo caminhadas para cada nó do grafo ref
        train_walks = []
        for nodeid in tqdm(self._G.nodes(), desc="- Sampling random walks"):
            walks = [self._sampler.sample_random_walk(nodeid, self._G) for _ in range(self._num_walks_per_node)]
            train_walks.extend(walks)

        # TODO: remover
        self.walk_lens = [len(walk) for walk in train_walks]

        # obtendo metadados para todas as caminhadas
        train_walks_with_metadata = []
        for walk in tqdm(train_walks, desc="- Adding metadata to walks"):
            walk_metadata = [self._get_node_metadata(nodeid) for nodeid in walk]
            train_walks_with_metadata.append(walk_metadata)
        
        train_walks_strs = []
        # convertendo caminhadas para strings json
        if self._use_json_serialization:
            for walk_metadata in tqdm(train_walks_with_metadata, desc="- Converting walks to JSON strings"):
                walk_json_str = json.dumps(walk_metadata, indent=None) # todo, precisa desse indent?
                train_walks_strs.append(walk_json_str)
        # conversao textual mais simples
        else:
            for walk_metadata in tqdm(train_walks_with_metadata, desc="- Converting walks to string format"):
                final_str = ""
                for event in walk_metadata:

                    event_str = EVENT_STR_TEMPLATE.format(
                        when=event['when'], 
                        what=event['what'],
                        who=event['who'],
                        why=event['why'],
                        where=event['where'],
                        how=event['how'],
                        id=event['event_id']
                    )
                    final_str += event_str + "\n"
                train_walks_strs.append(final_str.strip())
    
        # criando ds
        ds = Dataset.from_dict({"text": train_walks_strs})

        # aplicando pre-processamento (adição de tokens BOS e EOS)
        ds = ds.map(self._preprocessing_fn, batched=True, remove_columns=ds.column_names)

        # embaralhando e preparando splits
        ds = ds.shuffle()
        ds = ds.train_test_split(test_size=eval_pc)

        return (ds["train"], ds["test"])


class DPODatasetBuilder(BaseDatasetBuilder):

    def __init__(self, tokenizer: AutoTokenizer,
                       G_ref: nx.Graph,
                       bias: str = "none",
                       num_walks_per_node: int = 10,
                       walk_length: int = 5,
                       len_weighting: bool = False,
                       alpha: float = 2.0):

        super().__init__(tokenizer, G_ref)

        # parametros de random walks
        self._bias: str = bias
        self._num_walks_per_node: int = num_walks_per_node
        self._walk_length: int = walk_length

        # sampler
        self._sampler = RandomWalkSampler(
            bias=self._bias,
            num_walks_per_node=self._num_walks_per_node,
            walk_length=self._walk_length
        )

        # matriz de adj para consulta
        self._adj_mat = nx.adjacency_matrix(G_ref).toarray()

        # mapeamento de id da matriz de adj para id do no
        self._seqid_to_nodeid = {i: nodeid for i, nodeid in enumerate(self._G.nodes())}
        self._nodeid_to_seqid = {nodeid: i for i, nodeid in self._seqid_to_nodeid.items()}

        # configurando amostragem baseada no tamanho da caminhada
        if len_weighting:
            self._alpha: float = alpha # expoente para o peso das sub-caminhadas. valores maiores favorecem mais as sub-caminhadas mais longas.
            self._sample_sub_sequences = self._sample_sub_sequences_len_weighted

    def build_train_ds(self, train_edges: list[tuple], train_labels: list[bool]) -> Dataset:
        return

    def build_test_ds(self, test_edges: list[tuple], test_labels: list[bool]) -> Dataset:
        return
    
    def _is_edge_valid(self, edge: tuple[str, str]) -> bool:

        if self._G.has_edge(edge[0], edge[1]):
            return True
        else:
            return False


    def _generate_random_walks(self) -> list[list[str]]:
        # obtendo caminhadas para cada nó do grafo ref
        train_walks = []
        for nodeid in tqdm(self._G.nodes(), desc="- Sampling random walks"):
            walks = []
            for _ in range(self._num_walks_per_node):
                walk = self._sampler.sample_random_walk(nodeid, self._G)
                if len(walk) > 1: # ignorar caminhadas de tam 1 (apenas o nó âncora)
                    walks.append(walk)
            # ignorar caminhadas de tam 1 (apenas o nó âncora)
            train_walks.extend(walks)
        
        return train_walks
    
    def _sample_sub_sequences_len_weighted(self, walk: list[str]) -> list[str]:
        # sample a random subsequence of a random walk, with a weighting that favors longer sub-sequences
        walk_len = len(walk)
        if walk_len > 2:
            lengths = np.arange(1, walk_len) # pesos crescentes para sub-sequências mais longas
            weights = lengths ** self._alpha # aumentar a diferença entre os pesos
            weights = weights / weights.sum() # normalizar para obter probabilidades
            cutoff_id = np.random.choice(lengths, p=weights) # amostrar o ponto de corte com base nos pesos
        else:
            cutoff_id = 1
        
        sampled_walk = walk[:cutoff_id]

        return sampled_walk
    
    def _sample_sub_sequences(self, walk: list[str]) -> list[str]:
        # sample a random subsequence of a random walk
        walk_len = len(walk)
        if walk_len > 2:
            cutoff_id = np.random.randint(1, walk_len)
        else:
            cutoff_id = 1
        
        sampled_walk = walk[:cutoff_id]

        return sampled_walk
    
    def _sample_non_neighbour(self, reference_node: str) -> str:

        seq_id = self._nodeid_to_seqid[reference_node]
        non_neighbors = np.argwhere(self._adj_mat[seq_id] == 0).reshape(-1)
        sampled_seq_id = np.random.choice(non_neighbors).item()
        
        return self._seqid_to_nodeid[sampled_seq_id]
    
    def _sample_neighbour(self, reference_node: str) -> str:

        seq_id = self._nodeid_to_seqid[reference_node]
        neighbors = np.argwhere(self._adj_mat[seq_id] > 0).reshape(-1)
        sampled_seq_id = np.random.choice(neighbors).item()

        return self._seqid_to_nodeid[sampled_seq_id]
    
    def _get_node_str(self, nodeid: str) -> str:

        event_metadata = self._get_node_metadata(nodeid)
        event_str = EVENT_STR_TEMPLATE.format(
            id=nodeid,
            when=event_metadata["when"],
            where=event_metadata["where"],
            who=event_metadata["who"],
            what=event_metadata["what"],
            why=event_metadata["why"],
            how=event_metadata["how"],
        )

        return event_str

    def _convert_walk_to_str(self, walk: list[str]) -> str:
        # converter uma caminhada (lista de nodeids) para string, usando o template de evento
        event_strs = []
        for nodeid in walk:
            event_str = self._get_node_str(nodeid)
            event_strs.append(event_str)
        walk_str = "\n".join(event_strs)

        return walk_str


    def build_train_ds(self, validate: bool = False) -> Dataset:
        """
        Builds a dataset for training a DPO model on the graph data, using random walks as inputs.
         Returns:
            A Hugging Face Dataset object containing the training data.
        """

        # obtendo caminhadas para cada nó do grafo ref
        train_walks = self._generate_random_walks()

        # amostrando subcaminhadas de tam variavel min 1 e maximo max_len - 1
        sampled_sub_walks = [self._sample_sub_sequences(walk) for walk in tqdm(train_walks, desc="- Sampling sub-sequences from walks")]

        prompts = sampled_sub_walks
        chosens = []
        rejecteds = []

        for prompt in prompts:
            last_node = prompt[-1]

            sampled_neighbour = self._sample_neighbour(last_node)
            sampled_non_neighbour = self._sample_non_neighbour(last_node)

            chosens.append(sampled_neighbour)
            rejecteds.append(sampled_non_neighbour)

        # armazenando para posterior consulta
        self._prompts = prompts
        self._chosens = chosens
        self._rejecteds = rejecteds
        
        if validate:
            print("- Validating chosen and rejected edges against G_train...")
            for walk, chosen, rejected in zip(prompts, chosens, rejecteds):
                last_node = walk[-1]
                assert self._is_edge_valid((last_node, chosen)), f"Chosen edge ({last_node}, {chosen}) not in G_train"
                assert not self._is_edge_valid((last_node, rejected)), f"Rejected edge ({last_node}, {rejected}) in G_train"

    
        # obtendo metadados para todos os prompts e suas extensões (chosens e rejecteds)
        prompts_with_metadata = []
        for prompt in tqdm(prompts, desc="- Obtaining prompts metadata"):
            prompt_metadata = [self._get_node_metadata(nodeid) for nodeid in prompt]
            prompts_with_metadata.append(prompt_metadata) 
        
        chosens_with_metadata = []
        for chosen in tqdm(chosens, desc="- Obtaining chosen nodes metadata"):
            chosen_metadata = self._get_node_metadata(chosen)
            chosens_with_metadata.append(chosen_metadata)   
        
        rejecteds_with_metadata = []
        for rejected in tqdm(rejecteds, desc="- Obtaining rejected nodes metadata"):
            rejected_metadata = self._get_node_metadata(rejected)
            rejecteds_with_metadata.append(rejected_metadata)   

        # serialização
        prompt_strs = [self._convert_walk_to_str(walk) for walk in prompts]
        chosen_strs = [self._get_node_str(nodeid) + self._tokenizer.eos_token for nodeid in chosens]
        rejected_strs = [self._get_node_str(nodeid) + self._tokenizer.eos_token for nodeid in rejecteds]

        # criando ds
        ds = Dataset.from_dict(
            {
                "prompt": prompt_strs,
                "chosen": chosen_strs,
                "rejected": rejected_strs,
            }
        )

        # embaralhando 
        ds = ds.shuffle()

        return ds
    
