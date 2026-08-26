from transformers import TrainerCallback
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForCausalLM
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import math
from unsloth import FastModel 
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix

class LinkPredictionEvalCallback(TrainerCallback):

    """Callback customizado para avaliação de link prediction durante o treinamento.
    Ele processa as predições do modelo, calcula métricas de avaliação e salva os resultados em um arquivo JSON.
    """ 

    def __init__(self, eval_ds: Dataset,
                       tokenizer: AutoTokenizer, 
                       eval_steps: int = 1,
                       eval_bs: int = 16, 
                       early_stopping: bool = True,
                       metric_for_best_model: str = "f1",
                       early_stopping_patience: int = 3,
                       log_file: str = "lp_metrics.json",
                       predictions_file: str = "lp_predictions.json"):

        self._labels: list[bool] = eval_ds["label"] # labels para avaliação
        self._eval_ds: Dataset = eval_ds.remove_columns(["label"]) # ja processado
        self._eval_steps: int = eval_steps
        self._eval_bs: int = eval_bs
        self._early_stopping: bool = early_stopping
        self._metric_for_best_model: str = metric_for_best_model
        self._early_stopping_patience: int = early_stopping_patience
        self._tokenizer: AutoTokenizer = tokenizer
        self._log_file: str = log_file
        self._predictions_file: str = predictions_file
        self._collator: DataCollatorForSeq2Seq = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding="longest",
            label_pad_token_id=tokenizer.pad_token_id
        )

        self._metrics: list[dict] = []
        self._predictions: list[dict] = []
        self._best_metric = -math.inf # supor que quanto maior melhor
        self._no_improvement_steps = 0

    
    def _parse_raw_preds(self, raw_preds: list[str]) -> list[bool]:
        
        # *EXTREMAMENTE* ingenuo, mas funciona
        final_preds = []
        for raw_pred in raw_preds:
            if "true" in raw_pred.lower():
                final_preds.append(True)
            elif "false" in raw_pred.lower():
                final_preds.append(False)
            else:
                final_preds.append(False) # ou algum valor padrão para casos ambíguos

        return final_preds

    def _extract_preds_from_generated_text(self, generated_texts: list[str]) -> list[str]:
        
        raw_preds = []
        for pred in generated_texts:
            pred_json_str = pred.split("\nmodel\n")[-1].strip() # extrai a parte gerada pelo modelo
            raw_preds.append(pred_json_str)

        return raw_preds
    
    def _run_evaluation(self, model: AutoModelForCausalLM, device: torch.device, eval_dl: DataLoader) -> list[str]:

        # ativando o modo de inferencia
        FastModel.for_inference(model)
        
        preds = []
        with torch.inference_mode():
            for batch in tqdm(eval_dl, desc="- Remaining batches", leave=False):
                
                batch = batch.to(device)
                
                generated_ids = model.generate(
                    **batch,
                    use_cache=False, # estoura a memoria :(
                    do_sample=False,
                    max_new_tokens=64 # TODO> expor esses params
                )
                
                # extraindo prompts
                input_lengths = batch["input_ids"].shape[1]
                generated_ids = generated_ids[:, input_lengths:]
                
                decoded_texts = self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                preds.extend(decoded_texts)

                # limpando memoria entre batches
                del generated_ids
                del batch

        # resetando o modelo para treinamento
        FastModel.for_training(model)
        
        return preds
    
    def _compute_metrics(self, labels: list[bool], preds: list[bool]) -> dict:

        accuracy = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='macro')
        recall = recall_score(labels, preds, average='macro')
        precision = precision_score(labels, preds, average='macro')
        cm = confusion_matrix(labels, preds).tolist()

        metrics = {
            "accuracy": accuracy,
            "f1": f1,
            "recall": recall,
            "precision": precision,
            "confusion_matrix": cm,
        }

        return metrics
    
    def _serialize_logs(self) -> None:

        # salvando os logs
        with open(self._log_file, 'w') as f:
            json.dump(self._metrics, f, indent=2)

        # salvando preds
        with open(self._predictions_file, 'w') as f:
            json.dump(self._predictions, f, indent=2)


    def on_step_end(self, args, state, control, **kwargs):
        
        # avalia-se a cada eval_steps
        if state.global_step % self._eval_steps != 0:
            return
        
        print("- Starting link prediction evaluation.")
        
        model = kwargs['model']
        device = model.device

        # loop de inferencia
        inference_dl = DataLoader(self._eval_ds, batch_size=self._eval_bs, collate_fn=self._collator)
        raw_preds = self._run_evaluation(model, device, inference_dl)
        
        # processando saidas brutas
        print(raw_preds[:5]) # debug
        final_preds = self._parse_raw_preds(raw_preds)

        # sumario de classificacao
        print(classification_report(self._labels, final_preds, digits=2))

        # calculando metricas individualmente
        curr_metrics = self._compute_metrics(self._labels, final_preds)
        # adicionando step
        curr_metrics["step"] = state.global_step

        # preparando logs de predição. TODO: no futuro remover esse log de predições. não é tao util
        curr_preds = {
            "raw_predictions": raw_preds,
            "step": state.global_step
        }

        self._metrics.append(curr_metrics)
        self._predictions.append(curr_preds)

        # salvando os logs
        self._serialize_logs()

        # early stopping
        if self._early_stopping:
            # metrica de monitoramento atual
            current_metric_value = curr_metrics[self._metric_for_best_model]

            # atualização do estado interno
            if current_metric_value > self._best_metric:
                self._best_metric = current_metric_value
                self._no_improvement_steps = 0
                # salva o modelo atual como o melhor modelo
                control.should_save = True
            else:
                self._no_improvement_steps += 1
                print(f"- No improvement in the last {self._no_improvement_steps} evaluations. (current: {current_metric_value:.4f}, best: {self._best_metric:.4f}).")

            if self._no_improvement_steps >= self._early_stopping_patience :
                print(f"- No improvement in {self._metric_for_best_model} for {self._early_stopping_patience} evaluations. Stopping training.")
                control.should_training_stop = True