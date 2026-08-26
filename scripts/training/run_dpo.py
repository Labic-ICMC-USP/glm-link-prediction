from unsloth import FastModel 
from unsloth import PatchDPOTrainer, get_chat_template
from argparse import ArgumentParser
from glm_based_event_analysis.utils.general import load_config
from glm_based_event_analysis.link_prediction.datasets import DPODatasetBuilder
from datasets import Dataset
import pickle
import json
import os
import networkx as nx
import numpy as np
from transformers import AutoTokenizer, AutoProcessor
from functools import partial
from transformers import set_seed
from glm_based_event_analysis.utils.graph import remove_edges_from_graph
from trl import DPOTrainer, DPOConfig
from transformers import EarlyStoppingCallback

import warnings
warnings.filterwarnings('ignore') # limpar logs


def create_train_dataset(graph_path: str,
                         test_edges_path: str,
                         tokenizer: AutoTokenizer,
                         dataset_config: dict,
                         rw_config: dict) -> Dataset:



    # carregando grafo e arestas de teste
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)

    with open(test_edges_path, 'r') as f:
        test_edges = json.load(f)

    G_train = remove_edges_from_graph(G, test_edges)

    # construcão dos ds
    builder = DPODatasetBuilder(
        G_ref=G_train,
        tokenizer=tokenizer,
        len_weighting=True,
        alpha=3.0,
        **rw_config,
    )

    train_ds = builder.build_train_ds(validate=dataset_config["validate"])

    # salvando datasets criados
    data_path = dataset_config["data_path"]

    # criar a pasta se nao existir
    os.makedirs(data_path, exist_ok=True)
    train_ds.to_parquet(f"{data_path}/train_ds.parquet")
  
    # treino e avaliação para clm
    return train_ds


def can_load_dataset(data_path: str) -> bool:
    # checa se arquivos de treino/teste existem
    return os.path.exists(f"{data_path}/train_ds.parquet")

def get_parser():

    parser = ArgumentParser(description="Run fine-tuning of a model based on a configuration file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--graph", type=str, required=True, help="Path to the 'pkl' file containing the graph data.")
    parser.add_argument("--test_edges", type=str, required=True, help="Path to the JSON file containing test edges for link prediction evaluation.")
    parser.add_argument("--only_show_data_stats", action='store_true', help="If set, the script will only show data statistics and exit without training.")
    parser.add_argument("--resume_from_ckpt", action='store_true', help="If set, resumes the training from the latest checkpoint.")
                  
    return parser


if __name__ == "__main__":

    PatchDPOTrainer()

    parser = get_parser()
    args = parser.parse_args()

    # carregando configs
    config = load_config(args.config, print_config=True)
  
    # configurando seed global
    set_seed(config["global_seed"])

    # isolando configs
    dataset_config = config["dataset_creation"]
    rw_config = config["random_walks"]
    unsloth_config = config["unsloth"]
    lora_config = config["lora"]
    training_config = config["training"]

    # carregando modelo e tokenizador
    model, _ = FastModel.from_pretrained(
        model_name = unsloth_config["model_name"],
        max_seq_length = unsloth_config["max_seq_length"], 
        full_finetuning = False, 
    )

    model = FastModel.get_peft_model(
                model,
                finetune_vision_layers     = False, # Turn off for just text!
                finetune_language_layers   = True,  # Should leave on!
                finetune_attention_modules = True,  # Attention good for GRPO
                finetune_mlp_modules       = True,  # Should leave on always!
                **lora_config
    )

    # por algum motivo o from pre trained nao recuperou a classe processor.
    tokenizer = AutoProcessor.from_pretrained(unsloth_config["base_model"])


    if can_load_dataset(dataset_config["data_path"]) and not dataset_config["recreate"]:
        # carrega datasets existentes
        print(f"- Found existing datasets at {dataset_config['data_path']}. Loading them instead of creating new ones.")
        train_ds = Dataset.from_parquet(f"{dataset_config['data_path']}/train_ds.parquet")

    else:
        # criando datasets
        train_ds = create_train_dataset(
            graph_path = args.graph,
            test_edges_path = args.test_edges,
            tokenizer = tokenizer.tokenizer,
            dataset_config = dataset_config,
            rw_config = rw_config
        )
  

    print(f"Train dataset size: {len(train_ds)}")

    print(train_ds)
    print(train_ds[0])

    # auxiliar na customização dos parametros de treino
    if args.only_show_data_stats:
        exit(0)


    # paradas do DPO
    train_args = DPOConfig(**training_config)

    # print(tokenizer)
    trainer = DPOTrainer(
        model = model,
        ref_model = None,
        args = train_args,
        beta = unsloth_config["beta"], # expor como param
        train_dataset = train_ds,
        tokenizer = tokenizer,
        max_length = unsloth_config["max_seq_length"],
        max_prompt_length = unsloth_config["max_seq_length"],
    )

    trainer_stats = trainer.train(resume_from_checkpoint = args.resume_from_ckpt)

    # print(trainer_stats)
