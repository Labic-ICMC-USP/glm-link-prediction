import math

dataset_examples = 102810
val_pct = 0.0
final_bs = 512 # o batch size corresponde a um step
warmup_steps_pct = 0.1
evaluate_after_n_examples = 20_000
epochs = 1

# determinando exemplos de treino/val
num_train_examples = math.ceil(dataset_examples * (1 - val_pct))
num_val_examples = math.ceil(dataset_examples * val_pct)

# determinando o numero de iterações por época
train_steps = math.ceil(num_train_examples / final_bs) * epochs
warmup_steps = math.ceil(warmup_steps_pct * train_steps)

# determinando a quantidade de steps ate aval
eval_steps = math.ceil(evaluate_after_n_examples / final_bs)

print("---- Train/eval information ----")
print(f"Total examples: {dataset_examples}")
print(f"Train examples: {num_train_examples}")
print(f"Validation examples: {num_val_examples}\n")

print("---- Parameters ----")
print(f"Total train steps per epoch: {train_steps}")
print(f"Train steps per epoch: {math.ceil(num_train_examples / final_bs)}")
print(f"Warmup steps: {warmup_steps} to a {warmup_steps_pct * 100}% of an epoch warmup")
print(f"Eval steps: {eval_steps}")