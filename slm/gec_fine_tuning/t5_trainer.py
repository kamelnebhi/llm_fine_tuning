#!/usr/bin/env python
# coding: utf-8

# =========================
# 1️⃣ Imports
# =========================
import os
import pandas as pd
import numpy as np
import torch
import evaluate
import sacrebleu
import boto3
import wandb

# =========================
# Imports for GEC F0.5
# =========================
#import errant
#import spacy
# Load spaCy model and ERRANT annotator
#nlp = spacy.load("en_core_web_sm")
#annotator = errant.load("en")

# Load exact match metric
exact_match = evaluate.load("exact_match")


from datasets import Dataset, DatasetDict
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

# Initialize WandB run
wandb.init(project='writing_gec', entity='knebhi')

# =========================
# 2️⃣ S3 & Model Parameters
# =========================
bucket_name = 'sagemaker-studio-oxs6vznjds'
key = 'writing_gec/data/writing_corrections_2024-12-17_17_23_54.csv'

MODELNAME = "t5-base"
PREFIX = "grammar: "

training_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Training device: {training_device}")

# =========================
# 3️⃣ Load CSV from S3
# =========================
s3_client = boto3.client('s3', region_name='us-east-1')
obj = s3_client.get_object(Bucket=bucket_name, Key=key)
df = pd.read_csv(obj['Body'])
df = df[:5000]  # optional limit rows
print(df.shape)

# Keep only relevant columns
df = df[["writing_id", "activity_instructions", "writing_content", "corrected_writing"]]

# Rename columns for HF dataset
df_hf = df.rename(columns={
    "writing_id": "_id",
    "activity_instructions": "task",
    "writing_content": "src",
    "corrected_writing": "tgt"
})

# Convert to HF Dataset
dataset = Dataset.from_pandas(df_hf, preserve_index=False)

# =========================
# 4️⃣ Train/Validation/Test Split
# =========================
split1 = dataset.train_test_split(test_size=0.3, seed=42)
split2 = split1['test'].train_test_split(test_size=0.5, seed=42)

ds_ft = DatasetDict({
    "train": split1['train'],
    "validation": split2['train'],
    "test": split2['test']
})

# =========================
# 5️⃣ Load Tokenizer & Model
# =========================
tokenizer = T5Tokenizer.from_pretrained(MODELNAME)
model = T5ForConditionalGeneration.from_pretrained(MODELNAME)

# =========================
# 6️⃣ Preprocessing
# =========================
def fix(example):
    """Remove numeric prefix before colon from source text."""
    parts = example["src"].split(":")
    example["src"] = " ".join(parts[1:]).strip()
    return example

ds_ft["train"] = ds_ft["train"].map(fix)

train_ds = ds_ft["train"]
val_ds = ds_ft["validation"]

def preprocess(example):
    """Tokenize inputs and labels."""
    input_text = PREFIX + example["src"]
    target_text = example["tgt"]
    model_inputs = tokenizer(input_text, truncation=True, padding="max_length", max_length=256)
    labels = tokenizer(target_text, truncation=True, padding="max_length", max_length=256)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = train_ds.map(preprocess, batched=False)
tokenized_val = val_ds.map(preprocess, batched=False)

# =========================
# 7️⃣ Metrics
# =========================
exact_match = evaluate.load("exact_match")

# =========================
# Compute metrics for Trainer
# =========================
def compute_metrics(eval_pred):
    """
    Compute metrics for seq2seq Trainer:
    - Exact match
    - BLEU
    """
    predictions, labels = eval_pred

    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels_clean = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels_clean, skip_special_tokens=True)

    # Exact match
    em = exact_match.compute(predictions=decoded_preds, references=decoded_labels)["exact_match"]

    # BLEU
    bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])
    bleu_score = bleu.score

    return {
        "exact_match": em,
        "bleu": bleu_score
    }


# =========================
# 8️⃣ Training
# =========================
training_args = Seq2SeqTrainingArguments(
    output_dir="./t5-grammar-corrector",
    do_eval=True,
    learning_rate=3e-5,
    per_device_train_batch_size=12,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    report_to="wandb",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

# =========================
# 9️⃣ Save Model Locally
# =========================
model_dir = "./t5-grammar-corrector"
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)

eval_results = trainer.evaluate()
print("📊 Evaluation Results:", eval_results)

# =========================
# 🔟 Upload Model to S3
# =========================
def upload_model_to_s3(local_dir, bucket_name, s3_prefix):
    s3 = boto3.client('s3')
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_dir)
            s3_path = os.path.join(s3_prefix, relative_path).replace("\\", "/")
            print(f"Uploading {local_path} to s3://{bucket_name}/{s3_path}")
            s3.upload_file(local_path, bucket_name, s3_path)

s3_prefix = "writing_gec/models/t5-grammar-corrector"
upload_model_to_s3(model_dir, bucket_name, s3_prefix)
