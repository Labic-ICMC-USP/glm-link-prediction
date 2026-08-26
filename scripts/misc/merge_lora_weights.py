from unsloth import FastModel
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to the directory containing the model and LoRA weights.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the directory where the merged model will be saved.")
    
    args = parser.parse_args()

    # Carregar o modelo e o tokenizer
    model, tokenizer = FastModel.from_pretrained(args.model)

    # Salvar o modelo com os pesos LoRA combinados
    model.save_pretrained_merged(args.output_dir, tokenizer, save_method="merged_16bit")