from unsloth import FastModel, get_chat_template
from transformers import AutoTokenizer
from argparse import ArgumentParser
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
from datasets import Dataset
from transformers import AutoProcessor, AutoTokenizer, DataCollatorForSeq2Seq
from glm_based_event_analysis.link_prediction.datasets import link_prediction_preprocessing
from glm_based_event_analysis.generation.formatting import format_event, is_event_valid

def padronize_events(events: list[dict]) -> list[dict]:
    """Função para padronizar a estrutura dos eventos gerados, garantindo que todos tenham as chaves necessárias e na ordem correta.'."""
    
    formatted_events = []
    for event in tqdm(events, desc="- Padronizing events", total=len(events)):
        # se o evento for valido, reordena as chaves.
        # caso o contrario, use como está
        if is_event_valid(event):
            event = format_event(event)
        else:
            # os Datasets do hf não lidam bem com campos faltando OU campos não padronizados
            # para evitar comportamentos estranhos, descartar eventos que não estejam no formato esperado.
            event = None

        formatted_events.append(event)
    
    return formatted_events

# tokenizando e preparando exemplos
def tokenizer_function(examples: dict, tokenizer: AutoTokenizer, max_seq_length: int):
    """Função de tokenização para o dataset de predição de link. Recebe um exemplo com um campo 'text' e retorna os ids tokenizados."""

    input_ids = tokenizer(
        examples["text"],
        max_length=max_seq_length, 
        add_special_tokens=True,
        truncation=True
    )

    return input_ids

def process_link_prediction_dataset(eval_ds: Dataset, tokenizer: AutoTokenizer, max_seq_length: int) -> Dataset:
    """
    Prepares the evaluation dataset for link prediction by tokenizing the text data and formatting it for use in training. This function applies a tokenization function to the 'text' field of the dataset, removes the original 'text' column, and sets the format of the dataset to 'torch' for compatibility with PyTorch models.
    Args:
        eval_ds (Dataset): The evaluation dataset containing a 'text' field that needs to be tokenized.
        tokenizer (AutoTokenizer): The tokenizer to be used for processing the text data in the dataset
        max_seq_length (int): The maximum sequence length for tokenization, which will be used to truncate the tokenized sequences if they exceed this length.
    Returns:
        Dataset: The processed evaluation dataset with tokenized input IDs and formatted for PyTorch.
    """

    # preparando ds para predição de link
    tok_func = partial(
        tokenizer_function, 
        tokenizer=tokenizer,
        max_seq_length=max_seq_length
        
    )

    eval_ds = eval_ds.map(tok_func, batched=True, remove_columns='text')
    eval_ds.with_format("torch")

    return eval_ds


def get_parser():
    parser = ArgumentParser(description="Evaluate generated events against ground truth links.")
    parser.add_argument("--generated_events", type=str, required=True, help="Path to the JSON file containing the generated events.")
    parser.add_argument("--model", type=str, required=True, help="Path to the fine-tunned model.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the evaluation on.")
    parser.add_argument("--bs", type=int, default=128, help="Batch size for evaluation.")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length for tokenization.")
    return parser

if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    # carregando eventos
    with open(args.generated_events, "r") as f:
        generated_events = json.load(f)

    # carregando modelo e processador
    model, tokenizer = FastModel.from_pretrained(args.model)
    FastModel.for_inference(model)
    print(f"Model and tokenizer initialized from {args.model}. ",  model.config.model_type)
    tokenizer = get_chat_template(tokenizer, model.config.model_type)
    
    #tokenizer = AutoTokenizer.from_pretrained(args.model)

    # extraindo dados para avaliação
    source_data = [event['source_event'] for event in generated_events]
    target_data = [event['generated_event'] for event in generated_events]

    # padronizando eventos de origem e alvo
    print("- Padronizing source and target events")
    source_data = padronize_events(source_data)
    target_data = padronize_events(target_data)

    # pct de eventos invalidos
    invalid_target = sum([1 for event in target_data if event is None])
    print(f"Invalid target events: {invalid_target}/{len(target_data)} ({invalid_target/len(target_data):.4f})")

    # criando o dataset de inferencia
    inference_ds = Dataset.from_dict({
        "source": source_data,
        "target": target_data,
        "label": [None] * len(generated_events) # placeholders
    })

    # processando
    pre_proc_func = partial(link_prediction_preprocessing, tokenizer=tokenizer, add_generation_prompt=True)
    inference_ds = inference_ds.map(pre_proc_func, batched=True, remove_columns=inference_ds.column_names)

    print("- Input example:")
    print(inference_ds[0]["text"])
 
    inference_ds = process_link_prediction_dataset(inference_ds, tokenizer.tokenizer if hasattr(tokenizer, 'tokenizer') else tokenizer, args.max_seq_length)

    # obtendo collator
    collator = DataCollatorForSeq2Seq(
        tokenizer= tokenizer.tokenizer if hasattr(tokenizer, 'tokenizer') else tokenizer,
        padding="longest",
        label_pad_token_id=tokenizer.pad_token_id
    )

    # loop de inferencia
    inference_dl = DataLoader(inference_ds, batch_size=args.bs, collate_fn=collator)

    # inferencia 
    raw_generated = []
    with torch.inference_mode():
        for batch in tqdm(inference_dl, desc="- Remaining batches", leave=False):

            batch = batch.to(args.device)
            
            generated_ids = model.generate(
                **batch,
                use_cache=False, # estoura a memoria :(
                do_sample=False,
                max_new_tokens=64 # TODO> expor esses params
            )

            # extraindo prompts
            input_lengths = batch["input_ids"].shape[1]
            generated_ids = generated_ids[:, input_lengths:]
            
            decoded_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            raw_generated.extend(decoded_texts)

            # restringindo o uso de memória GPU
            del generated_ids
            del batch
            torch.cuda.empty_cache()
    

    # extraindo rotulos
    labels = []
    for pred in raw_generated:
        if "true" in pred.lower():
            labels.append(True)
        elif "false" in pred.lower():
            labels.append(False)
        else:
            labels.append(None) # ou algum valor default para casos ambíguos

    # pct de links corretamente identificados
    correct = sum([1 for label in labels if label is True])
    total = len(labels)
    accuracy = correct / total
    print(f"Valid edges: {accuracy:.4f} ({correct}/{total})")

    # salvando resultados, sobrescrevendo o json original  com um novo campo
    for i, event in enumerate(generated_events):
        event['raw_pred'] = raw_generated[i]
        event['pred_label'] = labels[i]
    
    with open(args.generated_events, "w") as f:
        json.dump(generated_events, f, indent=2)




    

