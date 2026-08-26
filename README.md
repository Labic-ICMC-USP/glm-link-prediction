# GLM Based Event Analysis under Computational Constraints

Repository dedicated to storing code related to research on using Graph Language Models for event graph analysis, focusing on the link prediction task through the lens of Small Language Models.

# Installation 
```
pip install -e          # Edition mode.
pip install .           # Normal installation.
```

We ommit the instalattion of PyTorch, Transformers, vLLM for user customisation.

# Project Structure

This repository's structure includes:

- `README.md`
- `pyproject.toml` and `setup.py`
- `src/` main folder containing the code implementations
- `scripts/` stores auxiliary scripts used for data processing, training and inference
- `notebooks/` stores jupyter notebooks used for both development and data analysis

# Main

After installation, the following scripts are available for the experiments' reproducibility.

- `create_iptc_event_graph`
  - Combines data collection, pre-processing and graph creation in a single script.
  - Creates a **single graph** at time.

- `create_all_event_graphs`
  - Auxiliary script that runs the previous script for all target themes in an input folder. 

- `create_all_eval_splits`
  -  Auxiliary script that creates evaluation splits for all event graphs. 

- `evaluate_icl_link_prediction`
  - Runs all link prediction experiments using in-context learning using the inputs created by the previous scripts (graphs and evaluation splits).
  - Evaluates a **single model** at time.

The scripts' parameters can be viewed in detail using a "--h" argument. If not available after installation, relaunch the python package manager to update environment variables.

## License

This project is licensed under the **MIT License**.
See the [LICENSE](LICENSE) file for details.