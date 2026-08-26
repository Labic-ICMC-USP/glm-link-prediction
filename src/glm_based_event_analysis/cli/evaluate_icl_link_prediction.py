import os
from subprocess import run
from argparse import ArgumentParser
from pathlib import Path

TARGET_SCRIPT = "scripts/link_prediction/run_icl_link_prediction.py"

AVAILABLE_THEMES = [
    "conflict", "crime", "cybersecurity", 
    "disaster", "economy", "education", 
    "energy", "environment", "health",
    "labour", "politics", "science", 
    "society", "transport", "weather"
]

def get_parser() -> ArgumentParser:

    parser = ArgumentParser(description="Script that evaluates a LLM in all event graphs.")

    parser.add_argument("--graphs_folder", required=True, type=str, help="Path to a folder containig graphs created in '.pkl' format.")
    parser.add_argument("--labels_folder", type=str, required=True, help="Path to the folder containing the output files. The created labels are stored in JSON files. The folder is automatically created if not exists.")
    parser.add_argument("--results_folder", type=str, required=True, help="Path to the folder which will store the results.")

    # general parameters
    parser.add_argument("--model_config", type=str, required=True, help="Path to the YAML model configuration file. Must follow the format present in 'src/glm_based_event_analysis/config/link_prediction' ")
    parser.add_argument("--serialization_config", type=str, required=True, help="Path to the YAML serialization configuration file. Must follow the format present in 'src/glm_based_event_analysis/config/neighbourhood_serialization' ")


    return parser

def main():

    parser = get_parser()
    args = parser.parse_args() 

    graphs_folder_path = Path(args.graphs_folder)
    labels_folder_path = Path(args.labels_folder)
    results_folder_path = Path(args.results_folder)

    #TODO rodar para todos os modelos?

    for theme in AVAILABLE_THEMES:

        graph_path = graphs_folder_path / f"{theme}_event_graph.pkl"
        test_edges_path = labels_folder_path / f"{theme}_labels.json"
        results_path = results_folder_path / f"{theme}_results.json"

        script_args = [
            "python",
            TARGET_SCRIPT,
            f"--lp_config={args.model_config}",
            f"--serialization_config={args.serialization_config}",
            f"--graph={graph_path}",
            f"--edges={test_edges_path}",
            f"--output_path={results_path}"
        ]

        run(script_args)


if __name__ == "__main__":
    main()

# example: evaluate_icl_link_prediction --graphs_folder /exp_local/kenzosaki/validation/graphs --labels_folder /exp_local/kenzosaki/validation/labels --results_folder /exp_local/kenzosaki/validation/results --model_config /home/kenzosaki/repos/glm-based-event-analysis/src/glm_based_event_analysis/config/link_prediction/gemma4.yaml --serialization_config /home/kenzosaki/repos/glm-based-event-analysis/src/glm_based_event_analysis/config/neighbourhood_serialization/ollama.yaml