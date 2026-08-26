from glm_based_event_analysis.link_prediction.prompt_templates import *
import ollama
import requests
import os
import json
from abc import ABC, abstractmethod
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from pydantic import BaseModel

class LinkPredictionOutput(BaseModel):

    success: bool
    link: bool
    explanation: str
    raw_response: str

class BaseLinkPredictor:
    """Base class for event component extraction using a pre-trained language model. Defines basic shared methods and prompts for the 5W1H event extraction task."""
    
    def __init__(self, model_name: str, generation_params: dict | None = None):

        self._model_name: str = model_name
        self._generation_params: dict = generation_params if generation_params is not None else {} 
        self._system_prompt: str = self._get_system_prompt()
        self._system_prompt_with_neighbours: str = self._get_system_prompt_with_neighbours()
        self._debug : bool = False


    def _get_system_prompt(self) -> str:
        """Basic system prompt for the link prediction task."""
        return BASE_SYSTEM_PROMPT
    
    def _get_user_prompt(self, event_u: dict, event_v: dict) -> str:
        return BASE_USER_PROMPT.format(json.dumps(event_u, indent=2), json.dumps(event_v, indent=2))
    
    def _get_system_prompt_with_neighbours(self) -> str:

        return SYSTEM_PROMPT_WITH_NEIGHBOURS
    
    def _get_user_prompt_with_neighbours(self, event_u: dict, event_v: dict, neighbours_u: list[dict]) -> str:

        return USER_PROMPT_WITH_NEIGHBOURS.format(
            json.dumps(event_u, indent=2),
            json.dumps(event_v, indent=2),
            json.dumps(neighbours_u, indent=2)
        )
    
    @abstractmethod
    def _call(self, system_prompt: str, user_prompt: str) -> str:
        """Abstract method to call the language model with the given system and user prompts. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement the _call method.")

    def _parse_response(self, response: str | None) -> dict | None:
        """Parse the raw response from the model. Outputs a structured JSON when the response is valid,
           and returns None if the response is invalid or if there was an error."""

        if not response:
            return None
        
        # limpeza básica de estruturas que podem atrapalhar o parsing, 
        # incluindo: - blocos de código (```), - marcação de linguagem (json), - separadores (---) e caracteres de nova linha extras
        response = response.replace("```", "") \
                           .replace("json", "") \
                           .replace("---", "") \
                           .strip()

        try:
            parsed = json.loads(response)
            
        except json.JSONDecodeError as e:
            if self._debug:
                print(f"Failed to parse response as JSON: {response}")
                print(f"Error: {e}")
            parsed = None

        # validando se um dicionario foi gerado adequadamente
        if not isinstance(parsed, dict):
            parsed = None
        
        # checando se os campos estao corretos
        if parsed:
            link = parsed.get("link", False)
            explanation = parsed.get("explanation", "")
            if not isinstance(link, bool) or not isinstance(explanation, str):
                if self._debug:
                    print(f"Parsed response has invalid types: {parsed}")
                parsed = None
        
        return parsed

    def predict_link(self, event_u: dict, event_v: dict, neighbours_u: list[dict] | None = None) -> LinkPredictionOutput:
        """Predict the link between two events, optionally using neighbor information.
        Args:
            event_u (dict): Dictionary representing event u.
            event_v (dict): Dictionary representing event v.
            neighbours_u (list[dict] | None): List of neighbor dictionaries for event u, or None if no neighbors are used.
        Returns:
            LinkPredictionOutput: An object containing the link prediction result, explanation, and raw response.
        """

        if neighbours_u is not None:
            user_prompt = self._get_user_prompt_with_neighbours(event_u, event_v, neighbours_u)
            raw_response = self._call(self._system_prompt_with_neighbours, user_prompt)
        else:
            user_prompt = self._get_user_prompt(event_u, event_v)
            raw_response = self._call(self._system_prompt, user_prompt)

        parsed_response = self._parse_response(raw_response)

        if parsed_response is not None:
            output = LinkPredictionOutput(
                success=True,
                link=parsed_response.get("link", False),
                explanation=parsed_response.get("explanation", ""),
                raw_response=raw_response
            )
        else:
            output = LinkPredictionOutput(success=False, link=False, explanation="Failed to parse response", raw_response=raw_response if raw_response else "")

        return output

class OllamaLinkPredictor(BaseLinkPredictor):
    """Link predictor that uses Ollama's API to predict links between events."""
    
    def __init__(self, model_name: str,  generation_params: dict | None = None):
        super().__init__(model_name, generation_params)

    def _call(self, system_prompt: str, user_prompt: str) -> str:
        """Call the Ollama API with the given system and user prompts."""

        if self._debug:
            print("System prompt:", system_prompt)
            print("User prompt:", user_prompt)

        response = ollama.chat(
            model=self._model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options=self._generation_params
        )["message"]["content"]
        return response
    

class OpenRouterLinkPredictor(BaseLinkPredictor):

    def __init__(self, model_name: str = "google/gemini-2.5-flash-lite", generation_params: dict | None = None, max_tries: int = 20):
        
        super().__init__(model_name, generation_params)
        self.max_tries = max_tries

    def _call(self, system_prompt: str, user_prompt: str) -> str | None:
        
        if self._debug:
            print("System prompt:", system_prompt)
            print("User prompt:", user_prompt)

        for attempt in range(self.max_tries):
            try:
                response = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {os.getenv('OPENROUTER_KEY')}",
                    },
                    data=json.dumps({
                        "model": self._model_name,
                        "messages": [
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": user_prompt
                        }
                        ]
                    } | self._generation_params
                    )
                )

                # Check if the request was successful
                if response.status_code == 200:
                    # TODO: retornar mais infos, como tokens/s
                    return response.json()['choices'][0]['message']['content']
                else:
                    print(f"Attempt {attempt + 1} failed with status code {response.status_code}: {response.text}")
                
            except Exception as e:              
                print(f"Attempt {attempt + 1} failed with error: {e}")
        
        # If all attempts fail, return None
        return None

    def parallel_predict_links(self, events_u: list[dict], events_v: list[dict], neighbours_u: list[dict] | list[None] = None, n_jobs: int = 4) -> list[LinkPredictionOutput]:
        """Predict links in parallel using multiple threads for parallel API calls.
        Args:
            events_u (list[dict]): List of event dictionaries for node u.
            events_v (list[dict]): List of event dictionaries for node v.  
            neighbours_u (list[dict] | None): List of neighbor dictionaries for node u, or None if no neighbors are used.
            n_jobs (int): Number of parallel threads to use for API calls.
        Returns:
            list[LinkPredictionOutput]: List of predictions for each pair of events, where each prediction is a LinkPredictionOutput object.
        """
        
        # implementação antiga
        #print(f"- Predicting links in parallel with {n_jobs} threads.")
        #with ThreadPool(n_jobs) as pool:
        #    results = pool.starmap(self.predict_link, [(u, v, neighbours_u) for u, v, neighbours_u in zip(events_u, events_v, neighbours_u)])

        # preservando a ordem das respostas
        results = [None] * len(events_u)
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            future_to_index = {executor.submit(self.predict_link, u, v, neighbours): index for index, u, v, neighbours in zip(range(len(events_u)), events_u, events_v, neighbours_u)}
            for future in tqdm(as_completed(future_to_index), total=len(events_u), desc="Processing prompts"):
                index = future_to_index[future]
                response = future.result()
                results[index] = response

        return results


class vLLMLinkPredictor(BaseLinkPredictor):

    def __init__(self, model_name: str, tokenizer: str, generation_params: dict | None = None):
        
        super().__init__(model_name, generation_params)

        max_model_len = self._generation_params.get("max_model_len", 2048)  # definir um valor padrão para max_model_len
        # removendo dos args (não faz parte dos sampling params)
        if "max_model_len" in self._generation_params:
            del self._generation_params["max_model_len"]

        self._model: LLM = LLM(model_name, tokenizer=tokenizer, max_model_len=max_model_len)
        self._max_length: int = max_model_len - generation_params.get("max_tokens", 256) # tam max desconsiderando tokens de saida
        self._generation_params: dict = generation_params
        self._processor: AutoProcessor = AutoProcessor.from_pretrained(tokenizer)
    
    def _create_system_user_prompts(self, event_u: dict, event_v: dict, neighbours_u: list[dict] | None = None) -> list[dict]:
        
        # caso tenha vizinhos
        if neighbours_u is not None:
            system_prompt = self._system_prompt_with_neighbours
            user_prompt = self._get_user_prompt_with_neighbours(event_u, event_v, neighbours_u)
        else:
            system_prompt = self._system_prompt
            user_prompt = self._get_user_prompt(event_u, event_v)

        input_pair = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        if self._debug:
            print("System prompt:", system_prompt)
            print("User prompt:", user_prompt)

        return input_pair
    
    def _get_input_tokens(self, message_pair: list[dict]) -> str:
        """Tokenize a single system+user prompt pair for one example."""
        
        token_ids = self._processor.apply_chat_template(
            message_pair,
            tokenize=True,
            add_generation_prompt=True,
            #enable_thinking=False # alguns modelos usam essa flag
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

    def predict_links(self, events_u: list[dict], events_v: list[dict], neighbours_u: list[dict] | list[None] = None) -> list[LinkPredictionOutput]:
        """Predict links between events using the vLLM model. This method processes the inputs in batches and applies the template to generate predictions.
        Args:
            events_u (list[dict]): List of event dictionaries for node u.
            events_v (list[dict]): List of event dictionaries for node v.
            neighbours_u (list[dict] | None): List of neighbor dictionaries for node u, or None if no neighbors are used.
        Returns:
            list[LinkPredictionOutput]: List of predictions for each pair of events, where each prediction is a LinkPredictionOutput object.
        """

        # criando prompts para cada par de eventos (com vizinhos se fornecidos)
    
        messages_list = [
            self._create_system_user_prompts(event_u, event_v, neighbours) for event_u, event_v, neighbours in zip(events_u, events_v, neighbours_u)
        ]

        # aplicando o template de chat e truncando 
        prompts = [
             {"prompt_token_ids": self._get_input_tokens(message)} for message in messages_list
        ]

        # extraçao de componentes
        params = SamplingParams(**self._generation_params)
        outputs = self._model.generate(prompts, params, use_tqdm=True)

        # filtrando os outputs
        raw_outputs = []
        for output in outputs:
            generated_text = output.outputs[0].text
            raw_outputs.append(generated_text)
        
        # parsing
        results = []
        for raw_output in raw_outputs:
            parsed_response = self._parse_response(raw_output)
            if parsed_response is not None:
                output = LinkPredictionOutput(
                    success=True,
                    link=parsed_response.get("link", False),
                    explanation=parsed_response.get("explanation", ""),
                    raw_response=raw_output
                )
            else:
                output = LinkPredictionOutput(success=False, link=False, explanation="Failed to parse response", raw_response=raw_output if raw_output else "")

            results.append(output)

        return results

class LlamaCppLinkPredictor(BaseLinkPredictor):

    def __init__(self, generation_params: dict | None,
                       host: str = "localhost",
                       port: int = 8080,
                       max_retries: int = 5):
        
        super().__init__("local-model", generation_params)
        

        # endpoint do servidor
        self._server_url: str = f"http://{host}:{port}/v1/chat/completions"

        self._max_retries: int = max_retries
        self._generation_params: dict = generation_params

    def _call(self, system_prompt: str, user_prompt: str) -> str:

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
                    raw_response = response_data['choices'][0]['message']['content']
                    return raw_response
                
            except Exception as e:
                print(f"[WARN] Request {i + 1}/{self._max_retries} failed with exception: {e}")
                if i == self._max_retries - 1:
                    raise  # Re-levanta a exceção se todas as tentativas falharem

        # caso falhar em todas as tentativas, retorna None
        return None
    
    def parallel_predict_links(self, events_u: list[dict], events_v: list[dict], neighbours_u: list[dict] | list[None] = None, n_jobs: int = 4) -> list[LinkPredictionOutput]:
        """Predict links in parallel using multiple threads for parallel API calls.
        Args:
            events_u (list[dict]): List of event dictionaries for node u.
            events_v (list[dict]): List of event dictionaries for node v.  
            neighbours_u (list[dict] | None): List of neighbor dictionaries for node u, or None if no neighbors are used.
            n_jobs (int): Number of parallel threads to use for API calls.
        Returns:
            list[LinkPredictionOutput]: List of predictions for each pair of events, where each prediction is a LinkPredictionOutput object.
        """

        # preservando a ordem das respostas
        results = [None] * len(events_u)
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            future_to_index = {executor.submit(self.predict_link, u, v, neighbours): index for index, u, v, neighbours in zip(range(len(events_u)), events_u, events_v, neighbours_u)}
            for future in tqdm(as_completed(future_to_index), total=len(events_u), desc="Processing prompts"):
                index = future_to_index[future]
                response = future.result()
                results[index] = response

        return results

