from argparse import ArgumentParser
import os
from glm_based_event_analysis.graph_construction.graph_builder import EventGraphBuilder
from glm_based_event_analysis.utils.general import load_config
import pickle

def get_parser():

    parser = ArgumentParser(description="Generate event graph based on configuration file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input 'pkl' file containing events and their components.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the generated event graph (in 'pkl' format).")
    parser.add_argument("--overwrite", action="store_true", help="Whether to overwrite the output file if it already exists.")

    return parser

if __name__ == "__main__":
    
    parser = get_parser()
    args = parser.parse_args()

    # checa se o arquivo de saída já existe e se a opção de sobrescrever não foi ativada
    if os.path.exists(args.output_file) and not args.overwrite:
        print(f"Output file '{args.output_file}' already exists. Use --overwrite to overwrite it.")
        exit(0)

    # carregar configuração
    config = load_config(args.config, print_config=True)



    # carregando eventos e seus componentes
    with open(args.input_file, "rb") as f:
        events = pickle.load(f)
    
    print(f"- Loaded {len(events)} events from {args.input_file}")

    # criar o construtor de grafos e gerar o grafo de eventos
    builder = EventGraphBuilder(config)
    event_graph = builder.build_graph(events)

    print(f"- Generated event graph with {len(event_graph.nodes)} nodes and {len(event_graph.edges)} edges.")

    # checa se o diretorio raiz existe se nao cria
    if not os.path.exists(os.path.dirname(args.output_file)):
        os.makedirs(os.path.dirname(args.output_file))

    # salvar o grafo gerado
    with open(args.output_file, "wb") as f:
        pickle.dump(event_graph, f)