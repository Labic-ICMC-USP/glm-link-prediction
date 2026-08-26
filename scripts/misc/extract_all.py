import os 
from pathlib import Path

DATA_DIR = Path("/exp_local/kenzosaki/data/iptc/raw_news")
OUTPUT_DIR = Path("/exp_local/kenzosaki/data/iptc/processed")
SCRIPT = "../graph_creation/extract_5W1H_from_news.py"
CONFIG = "../../src/glm_based_event_analysis/config/extraction/llamacpp_gemma4.yaml"

for path in DATA_DIR.rglob("*.pkl"):
    input_folder = path.parent
    theme = path.parent.name

    print("---"*30)
    print("input folder:", input_folder)
    print("theme:", theme)


    os.system(f"python3 {SCRIPT} --config {CONFIG} --input_folder {input_folder} --output_file {OUTPUT_DIR / f'{theme}_extracted_events.pkl'}")
