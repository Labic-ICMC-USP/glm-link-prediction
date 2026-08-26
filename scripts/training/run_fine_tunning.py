from unsloth import FastModel 
from unsloth.chat_templates import get_chat_template
from argparse import ArgumentParser
from glm_based_event_analysis.utils.general import load_config
from glm_based_event_analysis.random_walks.neighbourhood_serialization import NeighbourhoodSerializer
from glm_based_event_analysis.link_prediction.datasets import LinkPredictionDatasetBuilder, LinkPredictionDatasetBuilderWithNeighbours
from glm_based_event_analysis.utils.sampling import sample_neg_edges
from glm_based_event_analysis.link_prediction.fine_tunning import LinkPredictionTrainer
from glm_based_event_analysis.utils.graph import remove_edges_from_graph
from datasets import Dataset
import pickle
import json
import os
import networkx as nx
import numpy as np
from transformers import AutoTokenizer
from functools import partial
from transformers import set_seed

import warnings
warnings.filterwarnings('ignore') # limpar logs

# tokenizando e preparando exemplos
def tokenizer_function(examples: dict, tokenizer: AutoTokenizer, max_seq_length: int):
    """Função de tokenização para o dataset de predição de link. Recebe um exemplo com um campo 'text' e retorna os ids tokenizados."""

    input_ids = tokenizer(
        examples["text"],
        max_length=max_seq_length, 
        add_special_tokens=True,
        truncation=True
    )

    return input_ids

# TODO: mover essa função para algum modulo
def sample_edges_for_training(G: nx.Graph, n_edges: int, include_negatives: bool = True) -> tuple[list[tuple], list[bool]]:
    """
    Samples edges from the training graph for fine-tuning. Can include both positive and negative edges based on the include_negatives flag.
    Args:
        G (nx.Graph): The training graph from which to sample edges.
        n_edges (int): The number of positive edges to sample. If include_negatives is True, the same number of negative edges will also be sampled.
        include_negatives (bool): includes negative edges if true.
    Returns:
        tuple[list[tuple], list[bool]]: A tuple containing a list of sampled edges (as tuples of node pairs) and a corresponding list of boolean labels indicating whether each edge is positive
    """
    
    # arestas positivas
    true_edges = list(G.edges(data=False))

    # caso usar todos as arestas, simplficar aqui
    if n_edges != len(true_edges):
        true_edge_sample_idxs = np.random.choice(np.arange(G.number_of_edges()), size=n_edges, replace=False)
        true_edge_samples = [true_edges[i] for i in true_edge_sample_idxs]
    else:
        true_edge_samples = true_edges

    true_edge_sample_labels = [True] * len(true_edge_samples)

    # arestas negativas
    if include_negatives:
        false_edge_samples = sample_neg_edges(G, n_edges)
        false_edge_sample_labels = [False] * len(false_edge_samples) 
    else:
        false_edge_samples = []
        false_edge_sample_labels = []   

    # combinando
    all_edge_samples = true_edge_samples + false_edge_samples
    all_edge_sample_labels = true_edge_sample_labels + false_edge_sample_labels

    return all_edge_samples, all_edge_sample_labels



def create_train_eval_test_datasets(graph_path: str,
                                    test_edges_path: str,
                                    tokenizer: AutoTokenizer,
                                    dataset_config: dict,
                                    serialization_config: dict = None) -> tuple[Dataset, Dataset, Dataset]:
    
    """
    Creates training, evaluation, and test datasets for link prediction fine-tuning. The training dataset is created by sampling edges from the training graph, while the evaluation and test datasets are created based on the provided test edges. The function also handles the serialization of the created datasets for future use.
    Args:
        graph_path (str): The file path to the pickled graph data.
        test_edges_path (str): The file path to the JSON file containing test edges for link prediction evaluation.
        tokenizer (AutoTokenizer): The tokenizer to be used for processing the text data in the datasets.
        dataset_config (dict): A dictionary containing configuration parameters for dataset creation.
        serialization_config (dict, optional): A dictionary containing configuration parameters for dataset serialization.
    Returns:
        tuple[Dataset, Dataset, Dataset]: A tuple containing the training dataset, evaluation dataset, and test dataset, all formatted as HF Datasets.
    """

    # carregando grafo e arestas de teste
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)

    with open(test_edges_path, 'r') as f:
        test_edges = json.load(f)
        # TODO: REMOVER
        # test_edges = test_edges[:50] 

    G_train = remove_edges_from_graph(G, test_edges)
    edges_to_sample = int(dataset_config["edge_sample_pct"] * G_train.number_of_edges())

    print(f"- Sampling {edges_to_sample}/{G_train.number_of_edges()} edges for training.")

    test_edges_ids = [(edge_info["u"], edge_info["v"]) for edge_info in test_edges] 
    test_labels = [edge_info["label"] for edge_info in test_edges]
    # convertendo para um simples true e false. TODO: fazer isto direto na criação no futuro
    test_labels = [True if label == 'positive' else False for label in test_labels]

    train_edges_sample, train_labels = sample_edges_for_training(
        G_train, 
        n_edges=edges_to_sample,
        include_negatives=dataset_config["neg_edges"]
    )

    # obtendo o construtor dos ds
    if not dataset_config["include_neighbours"]:
        builder = LinkPredictionDatasetBuilder(tokenizer, G)
    else:
        if serialization_config is None:
            raise ValueError("Serialization config must be provided if include_neighbours is True.")
        
        neigh_serializer = NeighbourhoodSerializer(serialization_config, G_train) # *deve* usar o grafo de treino
        builder = LinkPredictionDatasetBuilderWithNeighbours(tokenizer, G, neigh_serializer)

    train_ds = builder.build_train_ds(train_edges_sample, train_labels)
    test_ds = builder.build_test_ds(test_edges_ids, test_labels) 

    # serializando ds criados
    data_path = dataset_config["data_path"]
    train_ds.to_parquet(f"{data_path}/train_ds.parquet")
    test_ds.to_parquet(f"{data_path}/test_ds.parquet")

    # treino e teste (predição de link)
    return (train_ds, test_ds)

def process_link_prediction_dataset(eval_ds: Dataset, tokenizer: AutoTokenizer, max_seq_length: int) -> Dataset:
    """
    Prepares the evaluation dataset for link prediction by tokenizing the text data and formatting it for use in training. This function applies a tokenization function to the 'text' field of the dataset, removes the original 'text' column, and sets the format of the dataset to 'torch' for compatibility with PyTorch models.
    Args:
        eval_ds (Dataset): The evaluation dataset containing a 'text' field that needs to be tokenized.
        tokenizer (AutoTokenizer): The tokenizer to be used for processing the text data in the dataset
        max_seq_length (int): The maximum sequence length for tokenization, which will be used to truncate the tokenized sequences if they exceed this length.
    Returns:
        Dataset: The processed evaluation dataset with tokenized input IDs and formatted for PyTorch.
    """

    # preparando ds para predição de link
    tok_func = partial(
        tokenizer_function, 
        tokenizer=tokenizer,
        max_seq_length=max_seq_length
        
    )

    eval_ds = eval_ds.map(tok_func, batched=True, remove_columns='text')
    eval_ds.with_format("torch")

    return eval_ds

def can_load_datasets(data_path: str) -> bool:
    # checa se arquivos de treino/teste existem
    return all(os.path.exists(f"{data_path}/{filename}") for filename in ["train_ds.parquet", "test_ds.parquet"])

def get_parser():

    parser = ArgumentParser(description="Run fine-tuning of a model based on a configuration file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--graph", type=str, required=True, help="Path to the 'pkl' file containing the graph data.")
    parser.add_argument("--test_edges", type=str, required=True, help="Path to the JSON file containing test edges for link prediction evaluation.")
    parser.add_argument("--only_show_data_stats", action='store_true', help="If set, the script will only show data statistics and exit without training.")
    parser.add_argument("--serialization_config", type=str, default=None, help="Path to a YAML configuration file for dataset serialization. Must be specified if 'include_neighbours' is True in the main configuration.")
                        
    return parser


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    # carregando configs
    config = load_config(args.config, print_config=True)
    serialization_config = load_config(args.serialization_config) if args.serialization_config else None

    # configurando seed global
    set_seed(config["global_seed"])

    # isolando configs
    dataset_config = config["dataset_creation"]
    unsloth_config = config["unsloth"]
    lora_config = config["lora"]
    training_config = config["training"]
    evaluation_config = config["evaluation"]

    # carregando modelo e tokenizador
    model, tokenizer = FastModel.from_pretrained(
        model_name = unsloth_config["model_name"],
        max_seq_length = unsloth_config["max_seq_length"], 
        load_in_4bit = True, # por padrao, Qlora 4bit
        # unsloth_tiled_mlp = True,
        full_finetuning = False, 
    )

    # obtendo templates de chat
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = unsloth_config["model_family"],
    )

    if can_load_datasets(dataset_config["data_path"]) and not dataset_config["recreate"]:
        # carrega datasets existentes
        print(f"- Found existing datasets at {dataset_config['data_path']}. Loading them instead of creating new ones.")
        train_ds = Dataset.from_parquet(f"{dataset_config['data_path']}/train_ds.parquet")
        link_pred_ds = Dataset.from_parquet(f"{dataset_config['data_path']}/test_ds.parquet")

    else:
        # criando datasets
        train_ds, link_pred_ds = create_train_eval_test_datasets(
            graph_path = args.graph,
            test_edges_path = args.test_edges,
            tokenizer = tokenizer,
            dataset_config = dataset_config,
            serialization_config = serialization_config
        )

    # preparando ds para predição de link
    link_pred_ds = process_link_prediction_dataset(
        eval_ds = link_pred_ds,
        tokenizer = tokenizer.tokenizer, # o tokenizer do unsloth é um wrapper
        max_seq_length = unsloth_config["max_seq_length"]
    )


    print(f"Train dataset size: {len(train_ds)}")
    print(f"Link prediction dataset size: {len(link_pred_ds)}")

    # auxiliar na customização dos parametros de treino
    if args.only_show_data_stats:
        exit(0)


    trainer = LinkPredictionTrainer(
        model=model,
        tokenizer=tokenizer,
        lora_config=lora_config,    
        unsloth_config=unsloth_config,
        training_config=training_config,
        evaluation_config=evaluation_config
    )

    trainer.train(
        train_ds=train_ds, 
        eval_ds=link_pred_ds,
        resume_from_checkpoint=False # feio TODO: remover
    )

    # problema para amanha
    # sudo ubuntu-drivers autoinstall