from argparse import ArgumentParser
from glm_based_event_analysis.utils.general import load_config
from glm_based_event_analysis.link_prediction.evaluation import EventGraphSplitter
import json
import os
import pickle
from transformers import set_seed


def get_parser() -> ArgumentParser:
    
    parser = ArgumentParser(description="Create test edge set for link prediction.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML evaluation configuration file.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input graph file 'pkl' file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output file. Must be a 'json' file where the test edges will be saved.")
    parser.add_argument("--overwrite", action="store_true", help="Whether to overwrite existing output file.")

    return parser

if __name__ == "__main__":

    parser = get_parser()  
    args = parser.parse_args()

    # checando se o arquivo de resultados já existe, e pulando execução se não for para sobrescrever
    if os.path.exists(args.output_file) and not args.overwrite:
        print(f"Output file {args.output_file} already exists. Use --overwrite to overwrite it.")
        exit(0)

    config = load_config(args.config, print_config=True)

    # configurando seed
    set_seed(config["split"]["seed"])

    # carregando o grafo
    with open(args.input_file, "rb") as f:
        G = pickle.load(f)

    # obtendo arestas de teste
    splitter = EventGraphSplitter(G, config)
    test_edges = splitter.get_test_edges(return_similarities=True) # analisar similaridade será interessante depois


    # checando se o diretorio de saída existe, se não, criando
    if not os.path.exists(os.path.dirname(args.output_file)):
        os.makedirs(os.path.dirname(args.output_file))
    
    # serializando arestas criadas
    with open(args.output_file, "w") as f:
        json.dump(test_edges, f, indent=2)