from unsloth import FastModel
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding
from glm_based_event_analysis.generation.formatting import is_event_valid
from torch.utils.data import DataLoader
import torch
from datasets import Dataset
from tqdm import tqdm
import networkx as nx
import json
import re

class PreTrainedEventGenerator:

    def __init__(self, model: AutoModelForCausalLM,
                       tokenizer: AutoTokenizer):
        
        self._model: AutoModelForCausalLM = model
        self._tokenizer: AutoTokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer # caso esteja usando um modelo multimod
        self._pattern: re.Pattern = r"<event>(.*?)</event>"

        # compilando modelo

    @classmethod
    def from_unsloth_ckpt(cls, model_path: str,):

        model, tokenizer = FastModel.from_pretrained(model_path)
        FastModel.for_inference(model) # modo de aval

        return cls(model, tokenizer)
    
    def _parse_generation(self, generated_text: str) -> list[dict] | None:
        """Parse the generated text to extract structured event information. The expected format is:
            <event>
            ... 5W1H information ...
            </event>
        If the format is correct, it returns a list of dictionaries with the event information. If the format is incorrect or if there was an error during parsing, it returns None."""

        matches = re.findall(
            self._pattern,
            generated_text,
            re.DOTALL
        )

        if not matches:
            print(f"No events found in the generated text: {generated_text}")
            return None
        
        parsed = []
        for match in matches:
            event = {}
            for line in match.splitlines():
                line = line.strip()
                if line:
                    key = line.split(":")[0]
                    value = line.removeprefix(key + ":").strip() if ":" in line else "" # o texto pode conter ":" em seu conteúdo
                    key = key.strip().lower()

                    # renomeando campo de id
                    key = "event_id" if key == "id" else key
                    event[key] = value

                    # print(f"debug. line = {line}, key = {key}, value = {value}")

            # TODO: o comportamento atual é descartar eventos inválidos (que não seguem o formato 5W1H ou que tem componentes faltando)
            # talvez no futuro de para pensar em recuperar/reaproveitar eventos parcialmente válidos.  
            parsed.append(event if is_event_valid(event) else None)
        
        
        return parsed
    
    def batch_generation(self, prompts: list[str], bs: int = 8, device: torch.device = torch.device("cpu"), **generation_kwargs) -> list[dict | None]:
        """
        Generate events based on the given prompts. The generation is done in batches for efficiency. The generated text is then parsed to extract structured event information.
        Args:   
            prompts (list[str]): A list of input prompts for event generation.
            bs (int): Batch size for generation.
            device (torch.device): Device to run the model on.
            **generation_kwargs: Additional keyword arguments to pass to the model's generate method.
        Returns:
            list[dict]: A list of dictionaries containing the generated event information. If an event could not be parsed correctly, its corresponding entry in the list will be None.
        """

        generated_walks = []

        # preparando dataset de inferencia
        inference_ds = Dataset.from_dict({"prompt": prompts})

        def tokenizer_function(examples: dict):
            """Função de tokenização para o dataset de predição de link. Recebe um exemplo com um campo 'text' e retorna os ids tokenizados."""

            input_ids = self._tokenizer(
                examples["prompt"],
                max_length=generation_kwargs.get("max_length", 1024),
                add_special_tokens=False,
                truncation=True
            )

            return input_ids
        
        # tokenizando e preparando dataloader para inferência
        inference_ds = inference_ds.map(tokenizer_function, batched=True, remove_columns='prompt', desc="- Tokenizing prompts")
        inference_ds.set_format("torch")
        inference_dl = DataLoader(inference_ds, 
            batch_size=bs, 
            collate_fn=DataCollatorWithPadding(tokenizer=self._tokenizer, padding="longest", pad_to_multiple_of=8)
        )

        # inferencia
        with torch.inference_mode():
            for batch in tqdm(inference_dl, desc="- Generating events", leave=False):

                batch = batch.to(device)
                
                # geração das caminhadas como ids de embedding
                batch_emb_ids = self._model.generate(
                    **batch,
                    pad_token_id=self._tokenizer.pad_token_id,
                    **generation_kwargs
                )

                # extraindo prompts
                input_lengths = batch["input_ids"].shape[1]
                batch_emb_ids = batch_emb_ids[:, input_lengths:]

                batch_decoded = self._tokenizer.batch_decode(
                    batch_emb_ids.cpu(),
                    skip_special_tokens=True
                )

                generated_walks.extend(batch_decoded)

                # restringindo o uso de memória GPU
                del batch_emb_ids
                del batch
                torch.cuda.empty_cache()

        # extraindo componentes
        generated_events = [self._parse_generation(gen) for gen in generated_walks]
            
        return generated_events
  