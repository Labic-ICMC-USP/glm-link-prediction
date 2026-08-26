import os
from subprocess import run
from argparse import ArgumentParser
from pathlib import Path

TARGET_SCRIPT = "scripts/link_prediction/create_test_examples_for_link_prediction.py"

def get_parser() -> ArgumentParser:

    parser = ArgumentParser(description="Script that automatically creates evaluation splits from event graphs.")

    parser.add_argument("--input_folder", required=True, type=str, help="Path to a folder containig graphs created in '.pkl' format.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the folder containing the output files. The created labels are stored in JSON files. The folder is automatically created if not exists.")

    # general parameters
    parser.add_argument("--eval_config", type=str, required=True, help="Path to the YAML evaluation configuration file. Must follow the format present in 'src/glm_based_event_analysis/config/evaluation' ")

    return parser

def main():

    parser = get_parser()
    args = parser.parse_args() 

    input_folder_path = Path(args.input_folder)
    output_folder_path = Path(args.output_folder)

    # disparando coleta, extração de componentes e construção de cada grafo
    for path in input_folder_path.rglob("*.pkl"):
        theme = path.stem.split("_")[0]
        output_file = output_folder_path / f"{theme}_labels.json"

        script_args = [
            "python",
            TARGET_SCRIPT,
            f"--config={args.eval_config}",
            f"--input_file={path}",
            f"--output_file={output_file}",
        ]

        run(script_args)


if __name__ == "__main__":
    main()

# example: create_all_eval_splits --input_folder /exp_local/kenzosaki/validation/graphs --output_folder /exp_local/kenzosaki/validation/labels --eval_config /home/kenzosaki/repos/glm-based-event-analysis/src/glm_based_event_analysis/config/evaluation/link_prediction.yaml