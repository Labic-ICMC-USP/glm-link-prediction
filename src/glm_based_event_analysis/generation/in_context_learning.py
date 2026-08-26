from glm_based_event_analysis.generation.prompt_templates import *
from glm_based_event_analysis.generation.formatting import is_event_valid
import ollama
import requests
import os
import json
from abc import ABC, abstractmethod

class BaseEventGenerator:
    """Base class for event generation using a pre-trained language model."""
    
    def __init__(self, model_name: str, generation_params: dict | None = None):

        self._model_name: str = model_name
        self._generation_params: dict = generation_params if generation_params is not None else {} 
        self._system_prompt: str = self._get_system_prompt()
        self._system_prompt_with_neighbours: str = self._get_system_prompt_with_neighbours()
        self._debug : bool = False


    def _get_system_prompt(self) -> str:
        """Basic system prompt for the 5W1H event extraction task."""
        return BASE_SYSTEM_PROMPT
    
    def _get_user_prompt(self, event_u: dict) -> str:
        return BASE_USER_PROMPT.format(source_event=json.dumps(event_u, indent=2))
    
    def _get_system_prompt_with_neighbours(self) -> str:

        return SYSTEM_PROMPT_WITH_NEIGHBOURS
    
    def _get_user_prompt_with_neighbours(self, event_u: dict, neighbours_u: list[dict]) -> str:

        return USER_PROMPT_WITH_NEIGHBOURS.format(
            source_event=json.dumps(event_u, indent=2),
            neighborhood=json.dumps(neighbours_u, indent=2)
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
            # TODO: o comportamento atual é descartar eventos inválidos (que não seguem o formato 5W1H ou que tem componentes faltando)
            return parsed if is_event_valid(parsed) else None
        except json.JSONDecodeError as e:
            if self._debug:
                print(f"Failed to parse response as JSON: {response}")
                print(f"Error: {e}")
            return None

    def generate(self, source_event: dict, neighbours_u: list[dict] | None = None) -> dict | None:
        """Generates a new event based on the source event and its neighbours (if provided). If neighbours are provided, they will be included in the prompt to provide additional context for generation. The method returns a structured dictionary with the generated event information, or None if the generation or parsing failed.
        Args:
            source_event (dict): A dictionary containing the information of the source event.
            neighbours_u (list[dict] | None): An optional list of dictionaries, each containing information about a neighbouring event. If provided, this information will be included in the prompt to provide additional context for the generation. If not provided, the generation will be based solely on the source event.
        Returns:
            dict | None: A dictionary containing the generated event information if the generation and parsing were successful, or None if there was an error during generation or if the generated response could not be parsed correctly."""

        if neighbours_u is not None:
            user_prompt = self._get_user_prompt_with_neighbours(source_event, neighbours_u)
            response = self._call(self._system_prompt_with_neighbours, user_prompt)
        else:
            user_prompt = self._get_user_prompt(source_event)
            response = self._call(self._system_prompt, user_prompt)
        
        return self._parse_response(response)
    
class OllamaEventGenerator(BaseEventGenerator):
    """Event generator that uses Ollama's API."""
    
    def __init__(self, model_name: str,  generation_params: dict | None = None):
        super().__init__(model_name, generation_params)

    def _call(self, system_prompt: str, user_prompt: str) -> str:
        """Call the Ollama API with the given system and user prompts."""

        if self._debug:
            print(f"System prompt:\n{system_prompt}\n")
            print(f"User prompt:\n{user_prompt}\n")
            
        response = ollama.chat(
            model=self._model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options=self._generation_params
        )["message"]["content"]
        return response