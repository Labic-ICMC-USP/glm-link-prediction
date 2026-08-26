from gnews import GNews 
from multiprocessing import Pool
from datetime import datetime, timedelta
from dateutil import parser as dateparser
import os
from tqdm import tqdm
import pickle

def fetch_news_for_target_date(client: GNews, target_date: datetime, keyword: str, day_increment: int) -> list[dict]:

    # customizando as datas para cada processo
    client.start_date = target_date - timedelta(days=day_increment-1)
    client.end_date = target_date

    #print(f"- Fetching news for {keyword} from {client.start_date} to {client.end_date}")

    try: 
        news = client.get_news(keyword)
    except Exception as e:        
        print(f"Error fetching news for {keyword} on {target_date.strftime('%Y-%m-%d')}: {e}")
        news = []
        
    return news

class GNewsDataCollector:

    def __init__(self, config: dict):
        
        self.language: str = config.get("language", "en")
        self.country: str = config.get("country", "US")
        self.jobs: int = config.get("jobs", 10)
        self.day_increment: int = config.get("day_increment", 2)
        self.max_results: int = config.get("max_results", 100)
        self.start_date: datetime | None = config.get("start_date", None)
        self.end_date: datetime | None = config.get("end_date", None)
        self.max_news_per_keyword: int = config.get("max_news_per_keyword", None)

        # parsing dates
        self.start_date: datetime = dateparser.parse(self.start_date) if self.start_date else datetime.now()
        self.end_date: datetime = dateparser.parse(self.end_date) if self.end_date else datetime.now()

        # o incremento de dias deve ser ao menos 2 pois a busca nao pode iniciar e terminar no mesmo dia
        assert self.day_increment >= 2, "day_increment must be at least 2 to ensure a valid date range for news search."

    def _create_client(self):
        
        # Cria o cliente com parametros comuns para todas as buscas
        return GNews(
            language=self.language, 
            country=self.country, 
            max_results=self.max_results, 
        )

    def search(self, queries: list[str], output_folder: str = "collected_news") -> None:

        # checks if folder exists or creates it
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for query in queries:
            results = []
            print(f"- Searching for news related to: {query}")

            # reseting start dates and batches for each keyword
            curr_batch = []
            curr_date = self.end_date # TODO: refatorar para priorizar mais recentes?
            while curr_date >= self.start_date:
                

                curr_batch.append(curr_date)
                curr_date = curr_date - timedelta(days=self.day_increment)

                if len(curr_batch) == self.jobs:
                    # disparar processos
                    print(f"- Attempting to fetch news for {curr_batch[0].date()} to {curr_batch[-1].date()}")
                    
                    with Pool(processes=self.jobs) as pool:
                        curr_results = pool.starmap(fetch_news_for_target_date, [(self._create_client(), date, query, self.day_increment) for date in curr_batch])

                    # adding the use query for logs
                    compiled_results = []
                    for batch_results in curr_results:
                        for news in batch_results:
                            news['query'] = query
                        compiled_results.extend(batch_results)

                    results.extend(compiled_results)

                    curr_batch = []
                    
                    print(f"- Total news collected for {query} so far: {len(results)}. Saving intermediate results...")

                    with open(os.path.join(output_folder, f"{query}.pkl"), "wb") as f:
                        pickle.dump(results, f)

                    # checando se o limite de notícias por palavra-chave foi atingido
                    if self.max_news_per_keyword and len(results) >= self.max_news_per_keyword:
                        print(f"- Reached max news limit for {query}. Stopping collection for this query.")
                        curr_date = self.start_date
                        
        
