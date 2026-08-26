from glm_based_event_analysis.graph_construction.prompt_templates import SYSTEM_PROMPT, USER_PROMPT
import ollama
from abc import abstractmethod
from datetime import datetime
import json
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
import torch
from pydantic import BaseModel

class ExtractionOutput(BaseModel):
    success: bool
    keywords: str
    what: str
    when: str
    where: dict | list[dict] # TODO: talvez de problema no futuro
    who: str
    why: str
    how: str
    raw_response: str

class BaseEventComponentExtractor:
    """Base class for event component extraction using a pre-trained language model. Defines basic shared methods and prompts for the 5W1H event extraction task."""
    
    def _get_system_prompt(self) -> str:
        """Basic system prompt for the 5W1H event extraction task."""
        return SYSTEM_PROMPT
    
    def _get_user_prompt(self, title: str, description: str, url: str, published: datetime) -> str:
        """Construct the user prompt with the news article."""

        published = published.isoformat()

        return USER_PROMPT.format(
            title=title,
            description=description,
            url=url,
            published=published
        )
    
    def _parse_response(self, response: str) -> dict | None:
        """Parse the raw response from the model into a structured event dictionary."""

        # limpeza básica de estruturas que podem atrapalhar o parsing, 
        # incluindo: - blocos de código (```), - marcação de linguagem (json), - separadores (---) e caracteres de nova linha extras
        response = response.replace("```", "") \
                           .replace("json", "") \
                           .replace("---", "") \
                           .strip()

        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            print(f"Response was: {response}")
            return None
            
    @abstractmethod
    def extract_event_components(self, title: str, description: str, url: str, published: datetime) -> ExtractionOutput:
        """Extract event components from the news article."""
        raise NotImplementedError("This method should be implemented by subclasses.")


class OllamaLLM(BaseEventComponentExtractor):

    def __init__(self, model_name: str, generation_params: dict | None = None):
        super().__init__()
        self.model: str = model_name
        self.generation_params: dict = generation_params if generation_params is not None else {"temperature": 0.0, "seed": 2026} # defaults to deterministic generation
        self.system_prompt: str = self._get_system_prompt()

    def _call(self, system_prompt: str, user_prompt: str):

        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options=self.generation_params
        )
        return response["message"]["content"]

    def extract_event_components(self, title: str, description: str, url: str, published: datetime) -> ExtractionOutput:
        """
        Extract event components from the news article using the Ollama LLM.
        Returns the parsed response from the model, which should be a dictionary representing the event.
        The response is expected to strictly follow the JSON schema defined in the system prompt, with no additional text or explanations.
        """

        user_prompt = self._get_user_prompt(title, description, url, published)
        raw_response = self._call(self.system_prompt, user_prompt)
        parsed_response = self._parse_response(raw_response) # TODO: retornar o raw response no futuro

        # overwrite the date with the real one
        if parsed_response and "when" in parsed_response:
            parsed_response["when"] = published.isoformat()
        
        if parsed_response:
            output = ExtractionOutput(
                success=True,
                keywords=parsed_response.get("keywords", ""),
                what=parsed_response.get("what", ""),
                when=published.isoformat(), # usando a data real de publicação do artigo, não a inferida pelo modelo
                where=parsed_response.get("where", {}),
                who=parsed_response.get("who", ""),
                why=parsed_response.get("why", ""),
                how=parsed_response.get("how", ""),
                raw_response=raw_response
            )
        else:
            output = ExtractionOutput(
                success=False,
                keywords="",
                what="",
                when=published.isoformat(), # idem
                where={},
                who="",
                why="",
                how="",
                raw_response=raw_response
            )  

        return output

class vLLM(BaseEventComponentExtractor):

    def __init__(self, model_name: str, tokenizer: str, generation_params: dict):
        # extraindo tam max do conteto
        max_model_len = generation_params.get("max_model_len", 2048)  # definir um valor padrão para max_model_len
        # removendo dos args (não faz parte dos sampling params)
        del generation_params["max_model_len"]

        self._model: LLM = LLM(model_name, tokenizer=tokenizer, max_model_len=max_model_len) # TODO: expor mais parametros
        self._max_length: int = max_model_len - generation_params.get("max_tokens", 256) # tam max desconsiderando tokens de saida
        self._generation_params: dict = generation_params
        self._processor: AutoProcessor = AutoProcessor.from_pretrained(tokenizer)
        
    def _create_system_user_prompts(self, title: str, 
                                          description: str, 
                                          url: str, 
                                          published_date: datetime) -> list[dict]:
        
        system_prompt = self._get_system_prompt()
        user_prompt_fn = self._get_user_prompt(title, description, url, published_date)

        input_pair = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt_fn}
        ]

        return input_pair
    
    def _get_input_tokens(self, message_pair: list[dict]) -> str:
        """Tokenize a single system+user prompt pair for one example."""
        
        token_ids = self._processor.apply_chat_template(
            message_pair,
            tokenize=True,
            add_generation_prompt=True,
        )

        # para compatibilidade entre processadores
        if hasattr(token_ids, "input_ids"):
            # caso seja um tokenizador simples unimodal
            token_ids = token_ids.input_ids
        elif isinstance(token_ids, list):
            # caso seja um tokenizador multimodal
            token_ids = token_ids[0]
        
        # Truncate from the left: drop oldest tokens, preserve the generation prompt at the end
        if len(token_ids) > self._max_length:
            token_ids = token_ids[-self._max_length:]
        
        return token_ids
    
    def extract_event_components(self, titles: list[str],
                                       descriptions: list[str],
                                       urls: list[str],
                                       published_dates: list[datetime]) -> list[ExtractionOutput]:
        
        # preparando os prompts
        messages_list = [
            self._create_system_user_prompts(
                title=title,
                description=description,
                url=url,
                published_date=published_date
            ) for title, description, url, published_date in zip(titles, descriptions, urls, published_dates)
        ]

        # aplicando o template de chat e truncando 
        prompts = [
             {"prompt_token_ids": self._get_input_tokens(message)} for message in messages_list
        ]

        # extraçao de componentes
        params = SamplingParams(**self._generation_params)
        outputs = self._model.generate(prompts, params, use_tqdm=True)

        # filtrando os outputs
        raw_components = []
        for output in outputs:
            generated_text = output.outputs[0].text
            raw_components.append(generated_text)
        
        # parsing e sobrescrevendo campo when com a data real de publicação do artigo
        parsed_components = []
        for raw_component, published_date in zip(raw_components, published_dates):
            parsed_component = self._parse_response(raw_component)
            if parsed_component:
                output = ExtractionOutput(
                    success=True,
                    keywords=parsed_component.get("keywords", ""),
                    what=parsed_component.get("what", ""),
                    when=published_date.isoformat(), # usando a data real de publicação do artigo, não a inferida pelo modelo
                    where=parsed_component.get("where", {}),
                    who=parsed_component.get("who", ""),
                    why=parsed_component.get("why", ""),
                    how=parsed_component.get("how", ""),
                    raw_response=raw_component
                )
            else:
                output = ExtractionOutput(
                    success=False,
                    keywords="",
                    what="",
                    when=published_date.isoformat(),
                    where={},
                    who="",
                    why="",
                    how="",
                    raw_response=raw_component
                )

            parsed_components.append(output)
            
        return parsed_components
    

class LlamaCpp(BaseEventComponentExtractor):

    def __init__(self, generation_params: dict,
                       host: str = "localhost",
                       port: int = 8080,
                       max_retries: int = 5):
        
        # endpoint do servidor
        self._server_url: str = f"http://{host}:{port}/v1/chat/completions"

        self._max_retries: int = max_retries
        self._generation_params: dict = generation_params

        # debug
        self._debug = False

    def _call(self, system_prompt: str, user_prompt: str) -> dict | None:

        # preparando o payload para a requisição
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": "local-model",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            **self._generation_params
        }

        for i in range(self._max_retries):
            try:
                response = requests.post(self._server_url, headers=headers, json=payload)
                # checa o status da resposta e, em caso de sucesso, retorna os dados da resposta
                if response.status_code == 200:
                    response_data = response.json()
                    return response_data
                
            except Exception as e:
                print(f"[WARN] Request {i + 1}/{self._max_retries} failed with exception: {e}")

        # caso falhar em todas as tentativas, retorna None
        return None
    
    def extract_event_components(self, titles: list[str],
                                       descriptions: list[str],
                                       urls: list[str],
                                       published_dates: list[datetime],
                                       n_jobs: int = 16) -> list[ExtractionOutput]:
        
        # preparando prompts
        system_prompt = self._get_system_prompt()
        user_prompts = [self._get_user_prompt(title, description, url, date) for title, description, url, date in zip(titles, descriptions, urls, published_dates)]

        # preservando a ordem das respostas, mesmo com execução paralela
        raw_responses = [None] * len(user_prompts)
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            future_to_index = {executor.submit(self._call, system_prompt, user_prompt): i for i, user_prompt in enumerate(user_prompts)}
            for future in tqdm(as_completed(future_to_index), total=len(user_prompts), desc="Processing prompts"):
                index = future_to_index[future]
                raw_responses[index] = future.result()

        # parsing e sobrescrevendo campo when com a data real de publicação do artigo
        parsed_components = []
        for raw_response, published_date in zip(raw_responses, published_dates):
            # visualizando saida raw, caso especificado
            if self._debug: print(raw_response)

            response_text = raw_response["choices"][0]["message"]["content"] if raw_response else ""

            parsed_component = self._parse_response(response_text)
            if parsed_component:
                output = ExtractionOutput(
                    success=True,
                    keywords=parsed_component.get("keywords", ""),
                    what=parsed_component.get("what", ""),
                    when=published_date.isoformat(), # usando a data real de publicação do artigo, não a inferida pelo modelo
                    where=parsed_component.get("where", {}),
                    who=parsed_component.get("who", ""),
                    why=parsed_component.get("why", ""),
                    how=parsed_component.get("how", ""),
                    raw_response=response_text
                )
            else:
                output = ExtractionOutput(
                    success=False,
                    keywords="",
                    what="",
                    when=published_date.isoformat(),
                    where={},
                    who="",
                    why="",
                    how="",
                    raw_response=response_text
                )

            parsed_components.append(output)
            
        return parsed_components
