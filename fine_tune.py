import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
import json

# Charger la configuration
with open("config.json", "r") as f:
    config = json.load(f)

MODEL_NAME = config["model_name"]
DATASET_PATH = config["dataset_path"]
OUTPUT_DIR = config["output_dir"]

# Vérifier si un GPU est dispo
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Entraînement sur : {device.upper()}")

# Charger le modèle et le tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

# Charger le dataset
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
print(f"Dataset chargé : {len(dataset)} échantillons")

# Tokenisation des données
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Définition des paramètres d'entraînement
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=config["learning_rate"],
    per_device_train_batch_size=config["batch_size"],
    num_train_epochs=config["epochs"],
    weight_decay=config["weight_decay"],
    save_total_limit=2,
    push_to_hub=False,
    fp16=True if device == "cuda" else False
)

# Initialisation de l'entraîneur
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# Lancement du fine-tuning
print("Début du fine-tuning...")
trainer.train()

# Sauvegarde du modèle
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Fine-tuning terminé ! Modèle sauvegardé dans : {OUTPUT_DIR}")
