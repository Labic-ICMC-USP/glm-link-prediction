from argparse import ArgumentParser
import json
import os
import pickle
import networkx as nx
from glm_based_event_analysis.link_prediction.in_context_learning import OllamaLinkPredictor, OpenRouterLinkPredictor, vLLMLinkPredictor, LlamaCppLinkPredictor, LinkPredictionOutput
from glm_based_event_analysis.random_walks.neighbourhood_serialization import NeighbourhoodSerializer
from glm_based_event_analysis.utils.general import load_config
from glm_based_event_analysis.utils.graph import remove_edges_from_graph
from tqdm import tqdm
from transformers import set_seed
from sklearn.metrics import classification_report
from dotenv import load_dotenv
import numpy as np

def run_ollama_link_prediction(eval_edges: list[dict], 
                               G: nx.Graph, 
                               link_predictor: OllamaLinkPredictor, 
                               use_neighbours: bool = False,
                               neigh_serializer: NeighbourhoodSerializer | None = None) -> tuple[list[bool], list[LinkPredictionOutput], list[list[str]]]:

    labels = []
    preds = []
    walks = []

    for edge in tqdm(eval_edges, desc="LLM inference", total=len(eval_edges)):

        u = edge['u']
        v = edge['v']
        event_u = get_node_info(u, G)
        event_v = get_node_info(v, G)
        walk = []

        if use_neighbours and neigh_serializer:
            neighborhood_representation = neigh_serializer.get_neighbourhood_representation(u, return_json_str=False)
            # armazenando ids da caminhada
            walk = neighborhood_representation[0]["walk_ids"]
            neighborhood_representation = neighborhood_representation[0]["walk"] # pegando apenas a caminhada
            pred = link_predictor.predict_link(event_u, event_v, neighborhood_representation)
        else:
            pred = link_predictor.predict_link(event_u, event_v)  

        labels.append(edge['label'])
        preds.append(pred)
        walks.append(walk)


    return labels, preds, walks

def run_openrouter_link_prediction(eval_edges: list[dict],
                                  G: nx.Graph,
                                  link_predictor: OpenRouterLinkPredictor | LlamaCppLinkPredictor,
                                  use_neighbours: bool = False,
                                  neigh_serializer: NeighbourhoodSerializer | None = None,
                                  n_jobs: int = 4) -> tuple[list[bool], list[LinkPredictionOutput], list[list[str]]]:
    
    labels = []
    walks = []
    events_u = []
    events_v = []
    neighbourhoods_u = []

    for edge in tqdm(eval_edges, desc="LLM inference", total=len(eval_edges)):

            u = edge['u']
            v = edge['v']
            walk = []
            event_u = get_node_info(u, G)
            event_v = get_node_info(v, G)
    
            if use_neighbours and neigh_serializer:
                neighborhood_representation = neigh_serializer.get_neighbourhood_representation(u, return_json_str=False)
                walk = neighborhood_representation[0]["walk_ids"] # armazenando ids da caminhada
                neighborhood_representation = neighborhood_representation[0]["walk"] # pegando apenas a caminhada
            else:
                neighborhood_representation = None
            
            events_u.append(event_u)
            events_v.append(event_v)
            neighbourhoods_u.append(neighborhood_representation)
            labels.append(edge['label'])
            walks.append(walk)

    i = 0
    while i < 5:
        rand_id = np.random.randint(0, len(events_u))
        print(f"Example event_u:\n{events_u[rand_id]}\n")
        print(f"Example event_v:\n{events_v[rand_id]}\n")
        print(f"Example neighbourhood_u:\n{neighbourhoods_u[rand_id]}\n")
        print("----")
        i+=1

    preds = link_predictor.parallel_predict_links(events_u, events_v, neighbourhoods_u, n_jobs)

    return labels, preds, walks

def run_vllm_link_prediction(eval_edges: list[dict],
                             G: nx.Graph,
                             link_predictor: vLLMLinkPredictor,
                             use_neighbours: bool = False,
                             neigh_serializer: NeighbourhoodSerializer | None = None) -> tuple[list[bool], list[LinkPredictionOutput], list[list[str]]]:
    
    labels = []
    walks = []
    events_u = []
    events_v = []
    neighbourhoods_u = []

    for edge in tqdm(eval_edges, desc="LLM inference", total=len(eval_edges)):

            u = edge['u']
            v = edge['v']
            event_u = get_node_info(u, G)
            event_v = get_node_info(v, G)
    
            if use_neighbours and neigh_serializer:
                neighborhood_representation = neigh_serializer.get_neighbourhood_representation(u, return_json_str=False)
                walk = neighborhood_representation[0]["walk_ids"] # armazenando ids da caminhada
                neighborhood_representation = neighborhood_representation[0]["walk"] # pegando apenas a caminhada
            else:
                neighborhood_representation = None
            
            events_u.append(event_u)
            events_v.append(event_v)
            neighbourhoods_u.append(neighborhood_representation)
            walks.append(walk)

            labels.append(edge['label'])
    
    i = 0
    while i < 5:
        rand_id = np.random.randint(0, len(events_u))
        print(f"Example event_u:\n{events_u[rand_id]}\n")
        print(f"Example event_v:\n{events_v[rand_id]}\n")
        print(f"Example neighbourhood_u:\n{neighbourhoods_u[rand_id]}\n")
        print("----")
        i+=1

    preds = link_predictor.predict_links(events_u, events_v, neighbourhoods_u)

    return labels, preds, walks


def get_node_info(node_id: str, G: nx.Graph) -> dict:

    node_attr = G.nodes[node_id]

    # atributos na ordem especificada nos prompts
    return {
        "what": node_attr["what"],
        "who": node_attr["who"],
        "when": node_attr["when"].strftime("%Y-%m-%d-%H:%M:%S"),
        "where": node_attr["where"],
        "why": node_attr["why"],
        "how": node_attr["how"],
    }

def get_parser():

    parser = ArgumentParser(description="Run link prediction using in-context learning and serialized neighbourhoods.")
    parser.add_argument("--lp_config", type=str, required=True, help="Path to the link prediction configuration file.")
    parser.add_argument("--serialization_config", type=str, default=None, help="Path to the serialization configuration file.")
    parser.add_argument("--edges", type=str, required=True, help="Path to the JSON file containing the evaluation edges.")
    parser.add_argument("--graph", type=str, required=True, help="Path to the graph pkl file used for metadata extraction.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the link prediction results.")
    parser.add_argument("--overwrite", action="store_true", help="Whether to overwrite existing results file.")

    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # checando se o arquivo de resultados já existe, e pulando execução se não for para sobrescrever
    if os.path.exists(args.output_path) and not args.overwrite:
        print(f"Results file {args.output_path} already exists. Use --overwrite to overwrite it.")
        exit(0)

    # carregando parametros de inferencia
    lp_config = load_config(args.lp_config, print_config=True)

    # configurando seed
    global_seed = lp_config.get("seed", 2026)
    set_seed(global_seed)

    # se preciso, carregar parametros de serialização para vizinhança
    serialization_config = {}
    if lp_config['use_neighbours']:
        assert args.serialization_config is not None, "Serialization config must be provided if use_neighbours is True."
        print(f"- Using additional neighbourhood information for link prediction.")
        serialization_config = load_config(args.serialization_config, print_config=True)

    # carregando grafo com metadados e arestas de teste
    with open(args.graph, "rb") as f:
        G = pickle.load(f)

    with open(args.edges, "r") as f:
        eval_edges = json.load(f)


    print(f"- Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    print(f"- Loaded {len(eval_edges)} evaluation edges.")

    model_name = lp_config['model_name']
    gen_params = lp_config.get("generation_params", {})

    if lp_config['provider'] == 'ollama':
        # forçando o contexto ser maior
        # os.environ["OLLAMA_CONTEXT_LENGTH"] = "10_000"
        print(f"Ollama context length set to: {os.environ.get('OLLAMA_CONTEXT_LENGTH', "env var not set")}")
        link_predictor = OllamaLinkPredictor(model_name, gen_params)
    elif lp_config['provider'] == 'openrouter':
        load_dotenv() # carregando variáveis de ambiente do .env (necessário para autenticação com OpenRouter)
        link_predictor = OpenRouterLinkPredictor(model_name, gen_params)
    elif lp_config['provider'] == 'vllm':
        tokenizer = lp_config.get("tokenizer", model_name) # se tokenizer não for especificado, o vLLMLinkPredictor usará o nome do modelo para inferir o tokenizer
        link_predictor = vLLMLinkPredictor(model_name, tokenizer, gen_params)
    elif lp_config['provider'] == 'llamacpp':
        link_predictor = LlamaCppLinkPredictor(gen_params)
    else:
        raise ValueError(f"Unsupported provider: {lp_config['provider']}")
    
    #eval_edges = eval_edges[:100]
    #link_predictor._debug = True

    # criando serializador de vizinhança, se necessário
    neigh_serializer = None
    if lp_config['use_neighbours']:
        # criando grafo de treino (sem arestas de teste e nós alvo) para extração de vizinhança
        G_train = remove_edges_from_graph(G, eval_edges)
        neigh_serializer = NeighbourhoodSerializer(serialization_config, G_train)

    if lp_config["provider"] == "ollama":
        labels, preds, walks = run_ollama_link_prediction(eval_edges, G, link_predictor, lp_config['use_neighbours'], neigh_serializer)

    elif lp_config["provider"] == "openrouter" or lp_config["provider"] == "llamacpp":
        # suporta tanto o openrouter quanto o llamacpp, pois ambos usam a mesma função de execução paralela
        labels, preds, walks = run_openrouter_link_prediction(eval_edges, G, link_predictor, lp_config['use_neighbours'], neigh_serializer, n_jobs=lp_config.get("n_jobs", 4))

    elif lp_config["provider"] == "vllm":
        labels, preds, walks = run_vllm_link_prediction(eval_edges, G, link_predictor, lp_config['use_neighbours'], neigh_serializer)

    logs = {
        "outputs": [output.model_dump() for output in preds],
        "labels": labels,
        "context_walks": walks,
    }

    # checando se a pasta de saída existe, e criando se não existir
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    with open(args.output_path, "w") as f:
        json.dump(logs, f, indent=2)

    filtered_preds = []
    filtered_labels = []
    for output, label in zip(preds, labels):
        if output.success:
            filtered_preds.append(output.link)
            filtered_labels.append(label) 

    print(f"- Reporting classification metrics:")
    print(classification_report(filtered_labels, filtered_preds))