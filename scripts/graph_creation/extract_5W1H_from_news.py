from glm_based_event_analysis.graph_construction.event_component_extraction import OllamaLLM, vLLM, LlamaCpp
from glm_based_event_analysis.utils.general import load_config
from datetime import datetime
import pickle 
from tqdm import tqdm
from dateutil import parser as dateparser
from argparse import ArgumentParser
import pandas as pd
from pathlib import Path
import os

def run_extraction_via_ollama(news_articles: list[dict], 
                              extractor: OllamaLLM, 
                              save_every: int, 
                              output_file: str) -> list[dict]:

    print(f"- Starting event component extraction using Ollama backend. Will save intermediate results every {save_every} articles to {output_file}.")

    processed_events = []

    for i, curr_news in enumerate(tqdm(news_articles, desc="Extracting events from news articles")):

        title = curr_news["title"]
        description = curr_news["description"]
        url = curr_news["url"]
        try:
            published = dateparser.parse(curr_news["published date"])
        except Exception as e:
            published = datetime.now()  # fallback para data atual se parsing falhar

        output = extractor.extract_event_components(title, description, url, published)
        # verificando se a extração foi bem sucedida
        if output.success:
            # isolando componentes
            event_components = output.model_dump(exclude={"success", "raw_response"})
            event_components["raw_desc"] = description  # opcional: manter a descrição original para referência futura
            event_components["raw_title"] = title  # opcional: manter o título original para referência futura
            event_components["published date"] = published.strftime("%Y-%m-%d %H:%M:%S")  
            event_components["query"] = curr_news["query"]  # opcional: manter a url original para referência futura
            processed_events.append(event_components)

        if i % save_every == 0 and i > 0:
            # saving the current extracted events as a JSONL file
            with open(output_file, "wb") as f:
                pickle.dump(processed_events, f)
            tqdm.write(f"- Saved {len(processed_events)} extracted events to {output_file}")
    
    return processed_events

def run_extraction_via_vllm(news_articles: list[dict], 
                            extractor: vLLM | LlamaCpp, 
                            save_every: int, 
                            output_file: str) -> list[dict]:

    print(f"- Starting event component extraction using vLLM/LlamaCpp backend. Will save intermediate results every {save_every} articles to {output_file}.")

    titles = []
    descriptions = []
    urls = []
    published_dates = []

    # isolando os campos de interesse
    for news in news_articles:
        titles.append(news["title"])
        descriptions.append(news["description"])
        urls.append(news["url"])
        try:
            published = dateparser.parse(news["published date"])
        except Exception as e:
            published = datetime.now()  # fallback para data atual se parsing falhar

        published_dates.append(published)
    
    # extraindo componentes
    outputs = extractor.extract_event_components(
        titles=titles,
        descriptions=descriptions,
        urls=urls,
        published_dates=published_dates,

    )

    # filtrando apenas os eventos extraídos com sucesso e adicionando campos opcionais de referência
    processed_events = []
    for output, news in zip(outputs, news_articles):
        if output.success:
            output_dict = output.model_dump(exclude={"success", "raw_response"})
            output_dict["raw_desc"] = news["description"] 
            output_dict["raw_title"] = news["title"]
            output_dict["published date"] = news["published date"]
            output_dict["query"] = news["query"]
            processed_events.append(output_dict)

    return processed_events

def get_parser():
    parser = ArgumentParser(description="Extract 5W1H event components from news articles using a pre-trained language model.")

    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file for the extraction backend LLM.") 
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the folder containing input news articles (in 'pkl' format).")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output file for saving extracted event components (in 'pkl' format).")

    return parser

if __name__ == "__main__":

    parser = get_parser()

    args = parser.parse_args()

    # load config
    config = load_config(args.config, print_config=True)
    save_every = config.get("save_every", 10)  # default to saving every 10 articles if not specified
    save_path = config.get("save_path", "extracted_events.pkl")  # default save path if not specified

    # carregando dados
    data_path = args.input_folder

    news_articles = []
    for file in Path(args.input_folder).glob("*.pkl"):
        print(f"- Reading file: {file}")
        with open(file, "rb") as f:
            curr = pickle.load(f)
        news_articles.extend(curr)

    print(f"- Loaded {len(news_articles)} news articles for extraction.")

    # criando caminho para salvar resultados intermediários (se nao existir)
    if not os.path.exists(os.path.dirname(args.output_file)):
        os.makedirs(os.path.dirname(args.output_file))
        
    # preprocessamento inicial -> remoção de duplicados baseado no titulo
    df = pd.DataFrame(news_articles)
    df.drop_duplicates(subset=["title"], inplace=True)
    news_articles = df.to_dict(orient="records")

    print(f"- After removing duplicates, {len(news_articles)} news articles remain for extraction.")

    #news_articles = news_articles[:30]  # TODO: remover

    processed_events = []
    # checando se o arquivo de saída já existe para carregar resultados intermediários (se necessário)
    if os.path.exists(args.output_file):
        print(f"- Output file {args.output_file} already exists. Loading existing extracted events.")
        with open(args.output_file, "rb") as f:
            processed_events = pickle.load(f)
        print(f"- Loaded {len(processed_events)} previously extracted events from {args.output_file}.")

        # determinando o evento de onde continuar
        # precisa ser manual pois pode ter ocorrido falha na extração de componentes.
        last_processed_event_title = processed_events[-1]["raw_title"]
        last_index = next((i for i, news in enumerate(news_articles) if news["title"] == last_processed_event_title), None) # vai ser o id ou None

        if last_index is not None:
            print(f"- Resuming extraction from index {last_index + 1} (after last processed event: '{last_processed_event_title}').")
            news_articles = news_articles[last_index + 1:]  # continuar a partir do próximo evento
        else:
            print(f"- Warning: unable to find the last processed event '{last_processed_event_title}'. Starting extraction from the beginning.")
            processed_events = []  # resetar eventos 

    if len(processed_events) > 0:
        print(f"- Resuming extraction with {len(processed_events)} already processed events.")

    # criando o extractor baseado no backend escolhido e executando
    model_name = config["extraction_config"]["model_name"]
    generation_params = config["extraction_config"]["generation_params"]
    if config["backend"] == "ollama":
        extractor = OllamaLLM(model_name=model_name, generation_params=generation_params)
        events = run_extraction_via_ollama(news_articles, extractor, save_every, args.output_file)
    elif config["backend"] == "vllm":
        tokenizer_name = config["extraction_config"].get("tokenizer", model_name)  # usar o mesmo nome do modelo como tokenizer se não especificado
        extractor = vLLM(model_name=model_name, tokenizer=tokenizer_name, generation_params=generation_params)
        events = run_extraction_via_vllm(news_articles, extractor, save_every, args.output_file)
    elif config["backend"] == "llamacpp":
        extractor = LlamaCpp(generation_params=generation_params) # TODO: num futuro expor o endpoint para customização
        events = run_extraction_via_vllm(news_articles, extractor, save_every, args.output_file)
    else:
        raise ValueError(f"Unsupported backend specified in config: {config['backend']}")

    # combinando a eventos ja processados (se houver)
    processed_events.extend(events)
        
    # ultima serialização
    with open(args.output_file, "wb") as f:
        pickle.dump(processed_events, f)

