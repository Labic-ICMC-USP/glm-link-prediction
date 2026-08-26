from glm_based_event_analysis.graph_construction.news import GNewsDataCollector
from glm_based_event_analysis.utils.general import load_config
from argparse import ArgumentParser
import os, pickle

def get_parser():
    parser = ArgumentParser(description="Collect news articles based on specified keywords and date ranges using the GNews API.")

    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file for news collection.") 
    parser.add_argument("--output_folder", type=str, required=True, help="Path to save the collected news articles (in 'pkl' format).")

    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    config = load_config(args.config, print_config=True)
    print(f"- Saving collected news articles to: {args.output_folder}")

    collector = GNewsDataCollector(config["gnews"])
    queries = config["queries"]
    # Example usage - replace with actual keywords and date range

    collector.search(queries, output_folder=args.output_folder)

    print("News collection completed.")
