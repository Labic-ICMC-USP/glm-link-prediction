from argparse import ArgumentParser
import json
import os
import pickle
import networkx as nx
from glm_based_event_analysis.generation.in_context_learning import OllamaEventGenerator
from glm_based_event_analysis.generation.pre_trained import PreTrainedEventGenerator
from glm_based_event_analysis.generation.formatting import prepare_json_str, prepare_simple_str
from glm_based_event_analysis.utils.general import load_config
from glm_based_event_analysis.utils.graph import get_node_metadata
from tqdm import tqdm
from dotenv import load_dotenv
from transformers import set_seed

def generate_events_with_pre_trained_model(source_nodes: list[dict], G: nx.Graph, event_generator: PreTrainedEventGenerator, gen_config: dict) -> list[dict | None]:
    """
    Generates events based on the given source nodes using a pre-trained language model. The generation can optionally include information from neighbouring events to provide additional context. The generated text is parsed to extract structured event information, which is returned as a list of dictionaries. If an event could not be generated or parsed correctly, its corresponding entry in the list will be None.
    Args:
        source_nodes (list[dict]): A list of dictionaries, each containing information about a source node (event) and its metadata for which to generate new events.
        G (nx.Graph): The graph containing the source nodes and their metadata. This graph is used to extract information about the source nodes and their neighbours (if applicable) to include in the generation prompts.
        event_generator (PreTrainedEventGenerator): An instance of the PreTrainedEventGenerator class, which is responsible for generating events based on the input prompts.
        gen_config (dict): A dictionary containing configuration parameters for the generation process, such as whether to use neighbouring events for context, batch size for generation, device to run the model on, and any additional generation parameters.
    Returns:
        list[dict | None]: A list of dictionaries containing the generated event information. Each dictionary corresponds to a source node and contains the original source event information along with the generated event information. If an event could not be generated or parsed correctly for a source node, its corresponding entry in the list will be None.
    """

    print(f"- Using neighbourhoods for generation: {gen_config['use_neighbours']}")
    prompts = []
    for event in tqdm(source_nodes, desc="- Preparing inputs", total=len(source_nodes)):

        u = event['source_node']

        if gen_config['use_neighbours']:
            context = event["random_walks"][0] # pegando apenas uma das caminhadas geradas. TODO: investigar alternativas eventualmente
            context.append(u) # adicionando o próprio nó de origem ao contexto
            prompt = prepare_simple_str(context, G)
        else:
            prompt = prepare_simple_str([u], G)

        prompts.append(prompt)

    print(f"- Prompt examples:")
    for i in range(min(3, len(prompts))):
        print(f"{prompts[i]}") 
        print("-" * 10)
    
    raw_generated_events = event_generator.batch_generation(
        prompts=prompts,
        bs=gen_config.get("batch_size", 2),
        device=gen_config.get("device", "cpu"),
        **gen_config.get("generation_params", {})
    )

    generated_events = []
    for event, raw_gen in zip(source_nodes, raw_generated_events):
        # ignorando eventos não validos na serialização.
        # note que pode serializar menos eventos que os prompts iniciais
        if raw_gen is None: continue
        new_event = {
            "source_event": get_node_metadata(G, event['source_node']),
            "generated_event": raw_gen[0] if raw_gen else None # TODO: por enquanto, pegando apenas um evento
        }
    
        generated_events.append(new_event)
    
    return generated_events

def generate_events_with_frozen_llm(source_nodes: list[dict], G: nx.Graph, event_generator: OllamaEventGenerator, gen_config: dict) -> list[dict]:

    print(f"- Using neighbourhoods for generation: {gen_config['use_neighbours']}")
    generated_events = []

    for event in tqdm(source_nodes, desc="- Generating events", total=len(source_nodes)):

        u = event['source_node']
        event_u = get_node_metadata(G, u)
        
        # todo: usar caminhadas
   
        if gen_config['use_neighbours']:
            neighborhood_ids = event["random_walks"][0] # pegando apenas uma das caminhadas geradas. TODO: investigar alternativas eventualmente
            neighborhood_representation = [get_node_metadata(G, node_id) for node_id in neighborhood_ids]
            gen_event = event_generator.generate(event_u, neighborhood_representation)
        else:
            gen_event = event_generator.generate(event_u)

        # ignorando eventos não validos na serialização.
        # note que pode serializar menos eventos que os prompts iniciais
        if gen_event is None: continue

        generated_events.append({
            "source_event": event_u,
            "generated_event": gen_event
        })
    
    return generated_events

def get_parser():
    parser = ArgumentParser(description="Run event generation using a frozen LLM and serialized neighbourhoods.")
    parser.add_argument("--generation_config", type=str, required=True, help="Path to the event generation configuration file.")
    parser.add_argument("--source_nodes", type=str, required=True, help="Path to the JSON file containing the source nodes.")
    parser.add_argument("--graph", type=str, required=True, help="Path to the graph pkl file used for metadata extraction.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the event generation results.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for reproducibility.")
    parser.add_argument("--use_neighbours", action="store_true", help="Overwrites the use_neighbours setting in the generation config. Used for quick testing.")

    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    set_seed(args.seed)

    # carregando parametros de inferencia
    gen_config = load_config(args.generation_config, print_config=True)

    if args.use_neighbours:
        gen_config['use_neighbours'] = True
        print("- Overwriting config to use neighbours for generation.")

    # carregando grafo com metadados e nós de origem
    with open(args.graph, "rb") as f:
        G = pickle.load(f)

    with open(args.source_nodes, "r") as f:
        source_nodes = json.load(f)

    # para testes
    #source_nodes = source_nodes[:50]

    model_name = gen_config['model_name']
    gen_params = gen_config.get("generation_params", {})

    print(f"Initializing event generator with model: {model_name}")
    if gen_config['provider'] == 'ollama':
        # forçando o contexto ser maior
        # os.environ["OLLAMA_CONTEXT_LENGTH"] = "10_000"
        print(f"Ollama context length set to: {os.environ.get('OLLAMA_CONTEXT_LENGTH', "env var not set")}")
        event_generator = OllamaEventGenerator(model_name, gen_params)
        #event_generator._debug = True # ativando debug para imprimir prompts e respostas
    elif gen_config['provider'] == 'openrouter':
        load_dotenv() # carregando variáveis de ambiente do .env (necessário para autenticação com OpenRouter)
        raise NotImplementedError("OpenRouter provider is not yet implemented.")
    elif gen_config['provider'] == 'pre_trained':
        event_generator = PreTrainedEventGenerator.from_unsloth_ckpt(gen_config['model_name'])
    else:
        raise ValueError(f"Unsupported provider: {gen_config['provider']}")

    if gen_config['provider'] in ["ollama", "openrouter"]:
        generated_events = generate_events_with_frozen_llm(source_nodes, G, event_generator, gen_config)
    elif gen_config['provider'] == "pre_trained":
        generated_events = generate_events_with_pre_trained_model(source_nodes, G, event_generator, gen_config)
    else:
        raise ValueError(f"Unsupported provider: {gen_config['provider']}")

    with open(args.output_path, "w") as f:
        json.dump(generated_events, f, indent=2)