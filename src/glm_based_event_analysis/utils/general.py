import yaml 

def load_config(config_path: str, print_config: bool = False) -> dict:
    """Load the YAML configuration file and highlight its contents."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if print_config:
        print(f"Loaded configuration file ({config_path}):")
        for key, value in config.items():
            print(f"\t {key}: {value}")
    return config