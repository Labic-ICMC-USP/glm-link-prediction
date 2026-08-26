import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import dot_score
from tqdm import tqdm
from typing import Generator
import networkx as nx

class EventGraphSplitter:

    def __init__(self, G, config: dict):
        self._G = G

        self._nodeid2seqid: dict[str, int] = {node: i for i, node in enumerate(G.nodes())}
        self._seqid2nodeid: dict[int, str] = {i: node for node, i in self._nodeid2seqid.items()}

        # isolando configs
        split_config = config["split"]
        negative_examples_config = config["negative_examples"]
        embedding_config = config["embedding"]

        # treino e teste
        self._test_pct: float = split_config.get("test_pct", 0.2) # pct de nós reservados para teste
        self._true_edge_quantity: float = split_config.get("true_edge_quantity", 1.0) # quantos pct das arestas de teste serão usadas de fato. pode ser a qtd direto

        # negativos
        self._hard_semantic_negatives_pct: float = negative_examples_config.get("hard_semantic_pct", 0.4) # pct de negativos difíceis entre os negativos usados para avaliação
        self._temporal_plausible_negatives_pct: float = negative_examples_config.get("temporal_plausible_pct", 0.4) # pct de negativos temporalmente plausíveis entre os negativos usados para avaliação
        self._random_negatives_pct: float = negative_examples_config.get("random_pct", 0.2) # pct de negativos aleatorios
        self._max_temporal_window: int = negative_examples_config.get("max_temporal_window", 31) # janela de dias para considerar um negativo temporalmente plausível

        # codificacao e similaridade semantica
        self._embedding_model_name: str = embedding_config.get("model_name", 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        self._encoding_batch_size: int = embedding_config.get("batch_size", 128)
        self._device: str = embedding_config.get("device", "cpu")

        # iniciando atributos
        self._hard_semantic_negatives: list[tuple[str, str]] = None
        self._temporal_plausible_negatives: list[tuple[str, str]] = None
        self._random_negatives: list[tuple[str, str]] = None
        self._train_nodes: list[str] = None
        self._test_nodes: list[str] = None
        self._true_test_edges: list[tuple[str, str]] = None
        self._semantic_similarities: np.ndarray = None
        self._is_newer_mask: np.ndarray = None
        self._available_negatives: set[tuple[str, str]] = None

    def _sample_random_negative(self) -> Generator[tuple[str, str], None, None]:
        """Samples a random negative pair (u, v) where u is from the training set and v is from the test set, ensuring that there is no edge between them in the original graph."""

        assert self._available_negatives is not None, "Negative pairs not initialized. Call _get_all_negative_pairs() first."
        
        # amostragem aleatoria de negativos
        while len(self._available_negatives) > 0:
            u, v = self._available_negatives.pop() # remove um par de nós do conjunto de negativos disponíveis
            yield (u, v)

    def _get_train_test_nodes(self):
        """Split nodes into training and test sets based on the "when" attribute, reserving the *most recent* nodes for testing."""

        # ordenando nós com base no atributo "when"
        sorted_nodes = sorted(self._G.nodes(data=True), key=lambda x: x[1].get("when", 0), reverse=True)

        # separando nós de treino e teste
        n_test = int(len(sorted_nodes) * self._test_pct)

        # ingenuo, nos de teste podem ser vizinhos entre si o que causa problemas na amostragem de vizinhanca
        #test_nodes = [node for node, _ in sorted_nodes[-n_test:]] 
        #train_nodes = [node for node, _ in sorted_nodes[:-n_test]]

        # pegar nós recentes que não tenham mais vizinhos
        test_nodes = []
        for node, _ in sorted_nodes:
            if self._G.out_degree(node) == 0: # nó com zero vizinhos de saída, ou seja, sem arestas saindo dele
                test_nodes.append(node)
            if len(test_nodes) >= n_test:
                break

        self._train_nodes = [node for node, _ in sorted_nodes if node not in test_nodes]
        self._test_nodes = test_nodes

        return self
    
    def _get_true_test_edges(self):
        """Filter true edges for testing, ensuring that the origin (u node) is in the train set and the destination (v node) is in the test set."""

        assert self._test_nodes is not None, "Test nodes not defined. Call _get_train_test_nodes() first."
        
        # filtrando todos as arestas que terminamm em nós do conjunto de teste
        test_edges = [(u, v) for u, v in self._G.edges() if v in self._test_nodes]

        # filtrando apenas arestas que *não* conectam componentes conexas
        final_test_edges = []
        for u, v in test_edges:
            edge_type = self._G[u][v]["edge_type"]
            if edge_type != "component_bridge":
                final_test_edges.append((u, v))
        
        print(f"- Total available true edges {len(final_test_edges)}")
        
        # selecionando um subconjunto das arestas de teste com base na porcentagem especificada
        if isinstance(self._true_edge_quantity, float) and 0 < self._true_edge_quantity <= 1:
            n_true_edges = int(len(final_test_edges) * self._true_edge_quantity)
            test_edges_idx = np.random.choice(np.arange(len(final_test_edges)), size=n_true_edges, replace=False).tolist()
        elif isinstance(self._true_edge_quantity, int) and self._true_edge_quantity > 0:
            test_edges_idx = np.random.choice(np.arange(len(final_test_edges)), size=min(self._true_edge_quantity, len(final_test_edges)), replace=False).tolist()

        # recuperando as arestas verdadeiras selecionadas
        true_test_edges = [final_test_edges[i] for i in test_edges_idx]

        # armazenando as arestas verdadeiras selecionadas
        self._true_test_edges = true_test_edges

        return self

    def _compute_semantic_similarities(self) -> np.ndarray:
        """Computing semantic similarities between all nodes using a pre-trained model and cosine similarity."""

        # computar embeddings usando um modelo pré-treinado
        all_whats = [self._G.nodes[node].get("what", "") for node in self._G.nodes()]

        model = SentenceTransformer(self._embedding_model_name, device=self._device)
        embeddings = model.encode(all_whats, batch_size=self._encoding_batch_size, normalize_embeddings=True)

        # calcular similaridade semântica (cosine similarity)
        similarities = dot_score(embeddings, embeddings).cpu().numpy()

        self._semantic_similarities = similarities

        return self

    def _get_all_negative_pairs(self):
        """Generate all possible negative pairs (u, v) where u is from the training set and v is from the test set, ensuring that there is no edge between them in the original graph.
        TODO: this is extremely memory intensive for large graphs. The most efficient way is to sample randomly, however it is not compatible with the creation of hard semantic negatives, 
              which requires knowledge of the similarity of all possible pairs. A *possible solution* is to generate a large pool of random negatives, compute their similarities and sample
              the top-k most similar as hard negatives.
        """

        adj_mat = nx.adjacency_matrix(self._G, weight=None) # não precisamos dos pesos, apenas do indicativo de existência de aresta
        non_adj_mat = 1 - adj_mat.toarray() - np.eye(adj_mat.shape[0]) # matriz de adjacencia negativa (1 onde nao tem aresta, 0 onde tem aresta ou diagonal)

        train_seq_ids = [self._nodeid2seqid[node] for node in self._train_nodes]
        test_seq_ids = [self._nodeid2seqid[node] for node in self._test_nodes]


        all_pairs = []
        for u in train_seq_ids:
            for v in test_seq_ids:
                if non_adj_mat[u, v] == 1:
                    u_nodeid = self._seqid2nodeid[u]
                    v_nodeid = self._seqid2nodeid[v]
                    all_pairs.append((u_nodeid, v_nodeid))

        # shuffle para garantir aleatoriedade na seleção dos negativos
        np.random.shuffle(all_pairs)
        self._available_negatives = set(all_pairs)

        print(f"- Total available negative pairs: {len(all_pairs)}")

        return self


    def _create_hard_semantic_negatives(self):
        """Create hard semantic negatives by selecting negative pairs with the highest semantic similarity to the true edges."""

        # determinando a quantidade de hard negatives
        n_hard_negatives = int(len(self._true_test_edges) * self._hard_semantic_negatives_pct)

        # obtendo todas as arestas possíveis e suas similaridades semânticas
        all_neg_pairs = self._available_negatives
        all_pairs_and_sims = []
        for u, v in all_neg_pairs:
            u_seqid = self._nodeid2seqid[u]
            v_seqid = self._nodeid2seqid[v]
            sim = self._semantic_similarities[u_seqid, v_seqid]
            all_pairs_and_sims.append((u, v, sim))

        # ordenando com base na similaridade, maior para o menor
        all_pairs_and_sims.sort(key=lambda x: x[2], reverse=True)

        hard_negatives = []
        curr_hard_negatives = 0
        with tqdm(total=n_hard_negatives, desc="- Creating hard semantic negatives") as pbar:
            for u, v, sim in all_pairs_and_sims:
                hard_negatives.append((u, v)) # invertendo a ordem para manter a consistencia com as arestas verdadeiras (origem -> destino)
                curr_hard_negatives += 1
                pbar.update(1)
                if curr_hard_negatives >= n_hard_negatives:
                    break

        self._hard_semantic_negatives = hard_negatives
        
        # atualizando o conjunto de negativos disponíveis removendo os hard negatives selecionados
        self._available_negatives = self._available_negatives - set(hard_negatives)
        
        return self

    def _create_temporal_plausible_negatives(self):
        """Create temporal plausible negatives by selecting negative pairs that are temporally close based on the "when" attribute of the nodes."""
        
        n_temporal_negatives = int(len(self._true_test_edges) * self._temporal_plausible_negatives_pct)
        temporal_negatives = []
        with tqdm(total=n_temporal_negatives, desc="- Creating temporal plausible negatives") as pbar:
            while len(temporal_negatives) < n_temporal_negatives:
                neg = next(self._sample_random_negative())

                # checagem de temporalidade 
                when_u = self._G.nodes[neg[0]].get("when")
                when_v = self._G.nodes[neg[1]].get("when")
                day_diff = (when_v - when_u).days

                # ignorar se v antecede u (dif negativa)
                if (day_diff > 0) and (day_diff <= self._max_temporal_window):
                    temporal_negatives.append(neg)
                    pbar.update(1)

        self._temporal_plausible_negatives = temporal_negatives

        return self

    def _create_random_negatives(self):
        """Create random negatives by sampling randomly from the remaining available negative pairs."""
        
        n_random_negatives = int(len(self._true_test_edges) * self._random_negatives_pct)
        random_negatives = []
        with tqdm(total=n_random_negatives, desc="- Creating random negatives") as pbar:
            while len(random_negatives) < n_random_negatives:
                neg = next(self._sample_random_negative())
                random_negatives.append(neg)
                pbar.update(1)

            random_negatives.append(neg)
        
        self._random_negatives = random_negatives

        return self

    def _label_test_edges(self):
        """Labeling test edges with their respective categories (true edge, hard semantic negative, temporal plausible negative, random negative) based on their characteristics and the computed semantic similarities."""

        # arestas verdadeiras
        true_labels_similarities = []
        for u, v in self._true_test_edges:
            u_seqid = self._nodeid2seqid[u]
            v_seqid = self._nodeid2seqid[v]
            sim = self._semantic_similarities[u_seqid, v_seqid]
            true_labels_similarities.append(sim)

        self._true_labels_similarities = true_labels_similarities # para visualizar dpois

        # limiares de interesse
        easy_threshold = np.percentile(true_labels_similarities, 75)
        medium_threshold = np.percentile(true_labels_similarities, 50)
        hard_threshold = np.percentile(true_labels_similarities, 25)

        labeled_true = []
        for u, v in self._true_test_edges:
            u_seqid = self._nodeid2seqid[u]
            v_seqid = self._nodeid2seqid[v]
            sim = self._semantic_similarities[u_seqid, v_seqid]

            # determimando a dificuldade com base na similaridade semântica
            if sim >= easy_threshold:
                label_description = "easy true edge (above 75th percentile similarity)"
            elif sim >= medium_threshold:
                label_description = "medium true edge (between 50th and 75th percentile similarity)"
            elif sim >= hard_threshold:
                label_description = "medium true edge (between 25th and 50th percentile similarity)"
            else:
                label_description = "hard true edge (below 25th percentile similarity)"

            labeled_true.append(
                {
                    "u": u,
                    "v": v,
                    "label": True,
                    "label_description": label_description
                }
            )

        # arestas negativas
        labeled_negatives = []
        for u, v in self._hard_semantic_negatives:
            labeled_negatives.append(
                {
                    "u": u,
                    "v": v,
                    "label": False,
                    "label_description": "hard semantic negative"
                }
            )
        for u, v in self._temporal_plausible_negatives:
            labeled_negatives.append(
                {
                    "u": u,
                    "v": v,
                    "label": False,
                    "label_description": "temporal plausible negative"
                }
            )
        for u, v in self._random_negatives:
            labeled_negatives.append(
                {
                    "u": u,
                    "v": v,
                    "label": False,
                    "label_description": "random negative"
                }
            )
        
        self._labeled_test_edges = labeled_true + labeled_negatives

        return self


    def get_test_edges(self, return_similarities: bool = False) -> list[dict]:
        """Main method to execute the entire pipeline of splitting the graph, generating positive and negative examples, and labeling the test edges for evaluation."""
        # TODO: note that the split are always 50% true edges and 50% negative edges, but the negative edges are subdivided into hard semantic, temporal plausible and random based on the specified percentages.
        #       A possible improvement is to allow customisation in this split as well.

        self._get_train_test_nodes(). \
             _get_true_test_edges(). \
             _compute_semantic_similarities(). \
             _get_all_negative_pairs(). \
             _create_hard_semantic_negatives(). \
             _create_temporal_plausible_negatives(). \
             _create_random_negatives(). \
             _label_test_edges()
        
        # adicionando similaridade nos dics.
        if return_similarities:
            for edge in self._labeled_test_edges:
                u_seqid = self._nodeid2seqid[edge["u"]]
                v_seqid = self._nodeid2seqid[edge["v"]]
                sim = self._semantic_similarities[u_seqid, v_seqid]
                edge["semantic_similarity"] = float(sim) # converter para float para garantir que seja serializável em json

        return self._labeled_test_edges
