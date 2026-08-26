import os
from subprocess import run
from argparse import ArgumentParser

DATA_COLLECTION_SCRIPT = "scripts/graph_creation/collect_news.py"
EXTRACTION_SCRIPT = "scripts/graph_creation/extract_5W1H_from_news.py"
GRAPH_CONSTRUCTION_SCRIPT = "scripts/graph_creation/generate_event_graph.py"

def get_parser():
    parser = ArgumentParser(description="Script that automatically collects news, extract 5W1H components and creates an event graph based on the collected news articles.")

    parser.add_argument("--collection_config", type=str, required=True, help="Path to the YAML configuration file for news collection.") 
    parser.add_argument("--extraction_config", type=str, required=True, help="Path to the YAML configuration file for 5W1H extraction.")
    parser.add_argument("--graph_config", type=str, required=True, help="Path to the YAML configuration file for graph construction.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the folder containing the output files including raw, processed (5W1H) and graph.")
    parser.add_argument("--file_prefix", type=str, default="news", help="Prefix for the output files.")

    # pulando etapas
    parser.add_argument("--skip_collection", action="store_true", help="Flag to skip the news collection step.")
    parser.add_argument("--skip_extraction", action="store_true", help="Flag to skip the 5W1H extraction step.")
    parser.add_argument("--skip_construction", action="store_true", help="Flag to skip the graph construction step.")

    # evitando sobreescrever arquivos
    parser.add_argument("--overwrite", action="store_true", help="Flag to allow overwriting existing output files. If not set, the script will check for existing files and skip steps if they already exist.")
    
    return parser

def main():

    parser = get_parser()
    args = parser.parse_args()

    # definindo saidas intermediarias e final
    news_output_folder = os.path.join(args.output_folder, "raw_news", args.file_prefix)
    extraction_output_file = os.path.join(args.output_folder, "processed", args.file_prefix + "_extracted_events.pkl")
    graph_output_file = os.path.join(args.output_folder, "graphs", args.file_prefix + "_event_graph.pkl")

    print(f"- Output folder for raw news: {news_output_folder}")
    print(f"- Output file for extracted events: {extraction_output_file}")
    print(f"- Output file for event graph: {graph_output_file}")

    # criando pasta de saida caso nao exista
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    if not args.skip_collection:
        if os.path.exists(news_output_folder) and not args.overwrite:
            print(f"[Warning]: The output folder for raw news '{news_output_folder}' already exists. Skipping news collection step to avoid overwriting existing data. Use --overwrite flag to allow overwriting.")
        else:
            print("-- Collecting news articles --")
            collection_args = [
                "python",
                f"{DATA_COLLECTION_SCRIPT}",
                f"--config={args.collection_config}",
                f"--output_folder={news_output_folder}"
            ]
            run(collection_args)

    if not args.skip_extraction:
        if os.path.exists(extraction_output_file) and not args.overwrite:
            print(f"[Warning]: The output file for extracted events '{extraction_output_file}' already exists. Skipping 5W1H extraction step to avoid overwriting existing data. Use --overwrite flag to allow overwriting.")
        else:
            print("-- Extracting 5W1H components --")
            extraction_args = [
                "python",
                f"{EXTRACTION_SCRIPT}",
                f"--config={args.extraction_config}",
                f"--input_folder={news_output_folder}",
                f"--output_file={extraction_output_file}"
            ]
            run(extraction_args)

    if not args.skip_construction:
        if os.path.exists(graph_output_file) and not args.overwrite:
            print(f"[Warning]: The output file for event graph '{graph_output_file}' already exists. Skipping graph construction step to avoid overwriting existing data. Use --overwrite flag to allow overwriting.")
        else:
            print("-- Constructing event graph --")
            construction_args = [
                "python",
                f"{GRAPH_CONSTRUCTION_SCRIPT}",
                f"--config={args.graph_config}",
                f"--input_file={extraction_output_file}",
                f"--output_file={graph_output_file}"
            ]
            run(construction_args)


if __name__ == "__main__":
    main()

# python3 create_iptc_event_graph.py --collection_config ../../glm-based-event-analysis/src/glm_based_event_analysis/config/collection/politics_iptc.yaml --extraction_config ../../glm-based-event-analysis/src/glm_based_event_analysis/config/extraction/ai_news.yaml --graph_config ../../glm-based-event-analysis/src/glm_based_event_analysis/config/graph_construction/ai_news.yaml --output_folder /exp_local/kenzosaki/data/iptc --file_prefix politics