from unsloth import FastModel 
from argparse import ArgumentParser
from glm_based_event_analysis.utils.general import load_config
from glm_based_event_analysis.link_prediction.datasets import CausalLMDatasetBuilder
from datasets import Dataset
import pickle
import json
import os
import networkx as nx
import numpy as np
from transformers import AutoTokenizer
from functools import partial
from transformers import set_seed
from glm_based_event_analysis.utils.graph import remove_edges_from_graph
from unsloth import UnslothTrainer, UnslothTrainingArguments
from transformers import EarlyStoppingCallback

import warnings
warnings.filterwarnings('ignore') # limpar logs


def create_train_eval_datasets(graph_path: str,
                               test_edges_path: str,
                               tokenizer: AutoTokenizer,
                               dataset_config: dict,
                               rw_config: dict) -> tuple[Dataset, Dataset]:



    # carregando grafo e arestas de teste
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)

    with open(test_edges_path, 'r') as f:
        test_edges = json.load(f)

    G_train = remove_edges_from_graph(G, test_edges)

    # construcão dos ds
    builder = CausalLMDatasetBuilder(
        G_ref=G_train,
        tokenizer=tokenizer,
        **rw_config,
    )

    train_ds, eval_ds = builder.build_train_eval_ds(dataset_config["val_pct"])

    # salvando datasets criados
    data_path = dataset_config["data_path"]
    train_ds.to_parquet(f"{data_path}/train_ds.parquet")
    eval_ds.to_parquet(f"{data_path}/eval_ds.parquet")
  
    # treino e avaliação para clm
    return (train_ds, eval_ds)


def can_load_datasets(data_path: str) -> bool:
    # checa se arquivos de treino/teste existem
    return all(os.path.exists(f"{data_path}/{filename}") for filename in ["train_ds.parquet", "eval_ds.parquet"])

def get_parser():

    parser = ArgumentParser(description="Run fine-tuning of a model based on a configuration file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--graph", type=str, required=True, help="Path to the 'pkl' file containing the graph data.")
    parser.add_argument("--test_edges", type=str, required=True, help="Path to the JSON file containing test edges for link prediction evaluation.")
    parser.add_argument("--only_show_data_stats", action='store_true', help="If set, the script will only show data statistics and exit without training.")
    parser.add_argument("--resume_from_ckpt", action='store_true', help="If set, resumes the training from the latest checkpoint.")
                  
    return parser


if __name__ == "__main__":

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
    model, tokenizer = FastModel.from_pretrained(
        model_name = unsloth_config["model_name"],
        max_seq_length = unsloth_config["max_seq_length"], 
        load_in_4bit = True, # por padrao, Qlora 4bit
        # unsloth_tiled_mlp = True,
        full_finetuning = False, 
    )

    if can_load_datasets(dataset_config["data_path"]) and not dataset_config["recreate"]:
        # carrega datasets existentes
        print(f"- Found existing datasets at {dataset_config['data_path']}. Loading them instead of creating new ones.")
        train_ds = Dataset.from_parquet(f"{dataset_config['data_path']}/train_ds.parquet")
        eval_ds = Dataset.from_parquet(f"{dataset_config['data_path']}/eval_ds.parquet")

    else:
        # criando datasets
        train_ds, eval_ds = create_train_eval_datasets(
            graph_path = args.graph,
            test_edges_path = args.test_edges,
            tokenizer = tokenizer,
            dataset_config = dataset_config,
            rw_config = rw_config
        )

  

    print(f"Train dataset size: {len(train_ds)}")
    print(f"Evaluation dataset size: {len(eval_ds)}")

    print(train_ds)
    print(train_ds[0])

    # auxiliar na customização dos parametros de treino
    if args.only_show_data_stats:
        exit(0)

    # usando o trainer diretamente por simplicidade
    # recomendaçõe https://unsloth.ai/blog/contpretraining
    model = FastModel.get_peft_model(
                model,
                finetune_vision_layers     = False, # Turn off for just text!
                finetune_language_layers   = True,  # Should leave on!
                finetune_attention_modules = True,  # Attention good for GRPO
                finetune_mlp_modules       = True,  # Should leave on always!
                **lora_config
    )

    train_args = UnslothTrainingArguments(**training_config)

    trainer = UnslothTrainer(
            model = model,
            tokenizer = tokenizer.tokenizer, # vou cometer um harakiri baiano. TODO: num futuro sempre checar se o tokenizer tem um atributo tokenizer,torch_empty_cache_steps: 20
            train_dataset = train_ds,
            eval_dataset = eval_ds, # avaliar apenas via callback customizado
            dataset_text_field = "text",
            dataset_kwargs = {
                 "skip_prepare_dataset": False,
            },
            max_seq_length = unsloth_config["max_seq_length"],
            dataset_num_proc = unsloth_config["num_proc"],
            args = train_args,
            processing_class = tokenizer,
    )

    # early stoping
    patience = training_config.get("save_total_limit", 2)
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience = patience)
    trainer.add_callback(early_stopping_callback)

    trainer_stats = trainer.train(resume_from_checkpoint = args.resume_from_ckpt)

    print(trainer_stats)

    # combinando os pesos Lora para criar o modelo final 
    model.save_pretrained_merged(f"{training_config['output_dir']}/merged", tokenizer, save_method = "merged_16bit",)
