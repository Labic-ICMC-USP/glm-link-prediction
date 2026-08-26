from unsloth import FastModel 
from unsloth.chat_templates import get_chat_template
from unsloth.chat_templates import train_on_responses_only
from transformers import AutoTokenizer, AutoModelForCausalLM
from glm_based_event_analysis.link_prediction.callbacks import LinkPredictionEvalCallback
from peft import PeftModel
from unsloth import UnslothTrainer, UnslothTrainingArguments
from datasets import Dataset
import math

class LinkPredictionTrainer:
    """Wrapper para o Trainer nativo do unsloth, para facilitar o processo de fine-tunning para predição de links."""

    def __init__(self, model: AutoModelForCausalLM, 
                       tokenizer: AutoTokenizer,
                       lora_config: dict,
                       unsloth_config: dict,
                       training_config: dict,
                       evaluation_config: dict,
                       ):

        # configurando LORA
        self._model: PeftModel = FastModel.get_peft_model(
                model,
                finetune_vision_layers     = False, # Turn off for just text!
                finetune_language_layers   = True,  # Should leave on!
                finetune_attention_modules = True,  # Attention good for GRPO
                finetune_mlp_modules       = True,  # Should leave on always!
                **lora_config
        )
        self._tokenizer: AutoTokenizer = tokenizer

        # armazenando configs
        self._training_config = SFTConfig(**training_config)
        self._unsloth_config = unsloth_config
        self._evaluation_config = evaluation_config

    def train(self, train_ds: Dataset, eval_ds: Dataset, resume_from_checkpoint: str | None = None):

        # definindo e configurando o trainer
        self._trainer = SFTTrainer(
            model=self._model,
            tokenizer=self._tokenizer,
            args=self._training_config,
            train_dataset=train_ds,
            eval_dataset=None, # avaliar apenas via o callback
        )

        # TODO: este o codigo que processa internamente os datasets consome *muita* RAM.
        # talvez salvar os datasets ja processados 
        self._trainer = train_on_responses_only(
            self._trainer,
            instruction_part=self._unsloth_config["instruction_part"],
            response_part=self._unsloth_config["response_part"],
            num_proc=self._unsloth_config["num_proc"],
        )

        # preparando callback de pred de link
        lp_callback = LinkPredictionEvalCallback(
            eval_ds=eval_ds,
            tokenizer=self._tokenizer.tokenizer, # o tokenizer do unsloth é um wrapper, entao acessamos o tokenizer original para o callback
            **self._evaluation_config
        )

        self._trainer.add_callback(lp_callback)


        trainer_stats = self._trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        return trainer_stats

    