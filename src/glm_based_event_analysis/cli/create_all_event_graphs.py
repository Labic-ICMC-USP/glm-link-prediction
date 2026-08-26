import os
from subprocess import run
from argparse import ArgumentParser
from pathlib import Path

TARGET_SCRIPT = "src/glm_based_event_analysis/cli/create_iptc_event_graph.py"

def get_parser() -> ArgumentParser:

    parser = ArgumentParser(description="Script that automatically creates event graphs for target themes. It sequentially collect news, extracts 5W1H components and creates event graphs.")

    parser.add_argument("--input_folder", required=True, type=str, help="Path to a folder containig configuration .yaml files for each target news' categories. The files must follow the structure present in the 'glm-based-event-analysis/src/glm_based_event_analysis/config' examples.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the folder containing the output files including raw, processed (5W1H) and graph.")

    # general parameters
    parser.add_argument("--extraction_config", type=str, required=True, help="Path to the YAML configuration file for 5W1H extraction. Must follow the format present in 'src/glm_based_event_analysis/config/extraction'.")
    parser.add_argument("--graph_config", type=str, required=True, help="Path to the YAML configuration file for graph construction. Must follow the format present in 'src/glm_based_event_analysis/config/graph_construction'.")

    return parser

def main():

    parser = get_parser()
    args = parser.parse_args() 

    input_folder_path = Path(args.input_folder)

    # disparando coleta, extração de componentes e construção de cada grafo
    for path in input_folder_path.rglob("*.yaml"):

        theme = path.stem.split("_")[0]

        script_args = [
            "python",
            TARGET_SCRIPT,
            f"--collection_config={path}",
            f"--extraction_config={args.extraction_config}",
            f"--graph_config={args.graph_config}",
            f"--output_folder={args.output_folder}",
            f"--file_prefix={theme}"
        ]

        run(script_args)


if __name__ == "__main__":
    main()

# example: create_all_event_graphs --input_folder /exp_local/kenzosaki/validation/themes --output_folder /exp_local/kenzosaki/validation/ --extraction_config /home/kenzosaki/repos/glm-based-event-analysis/src/glm_based_event_analysis/config/extraction/ollama.yaml --graph_config /home/kenzosaki/repos/glm-based-event-analysis/src/glm_based_event_analysis/config/graph_construction/ai_news.yaml