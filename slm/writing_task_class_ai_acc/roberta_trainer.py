#!/usr/bin/env python
# coding: utf-8

import os
import json
import numpy as np
import pandas as pd
import torch
from collections import Counter
from datasets import load_dataset, Dataset, DatasetDict, ClassLabel
from transformers import (
    AutoTokenizer,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import evaluate
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    cohen_kappa_score,
    classification_report,
)
from scipy.stats import pearsonr


import wandb

# Initialize a W&B run
wandb.init(project='writing_task_class_ai_acc', entity='knebhi')



def prepare_data(csv_path="data/acc_data.csv"):
    df = pd.read_csv(csv_path)
    #df = df[:3000]

    # Create the combined text field
    df['text'] = (
        "Prompt Level: " + df['level_title'].astype(str) +
        " [SEP] Prompt: " + df['activity_instructions'] +
        " [SEP] Response: " + df['student_submission']
    )

    df = df[["text", "task_id", "level_title", "majority_value"]]
    df = df.rename(columns={'majority_value': 'label'})
    df.dropna(subset=['label', 'text'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    ds = Dataset.from_pandas(df)

    # Define label feature with classes
    new_features = ds.features.copy()
    new_features["label"] = ClassLabel(names=[0, 1, 2, 3, 4, 5])
    ds = ds.cast(new_features)

    # Split dataset: train / test+validation
    train_test_ds = ds.train_test_split(test_size=0.20, seed=20)
    test_valid_split = train_test_ds['test'].train_test_split(test_size=0.5, seed=20)

    dataset = DatasetDict({
        'train': train_test_ds['train'],
        'test': test_valid_split['train'],          # test set
        'validation': test_valid_split['test']      # validation set
    })

    # Save datasets as JSONL
    os.makedirs("data", exist_ok=True)
    for split_name in ['train', 'test', 'validation']:
        save_split_to_jsonl(dataset[split_name], f"data/{split_name}.jsonl")

    return dataset


def save_split_to_jsonl(dataset_split, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for record in dataset_split:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    metrics = {
        "accuracy": float('nan'),
        "precision": float('nan'),
        "recall": float('nan'),
        "f1": float('nan'),
        "cohen_kappa": float('nan'),
        "pearson_corr": float('nan'),
        "classification_report": {}
    }

    try:
        if len(labels) >= 2 and len(set(labels)) > 1 and len(set(predictions)) > 1:
            metrics["accuracy"] = accuracy_score(labels, predictions)
            metrics["precision"], metrics["recall"], metrics["f1"], _ = precision_recall_fscore_support(
                labels, predictions, average="weighted", zero_division=0
            )
            metrics["cohen_kappa"] = cohen_kappa_score(labels, predictions, weights="quadratic")
            metrics["pearson_corr"], _ = pearsonr(labels, predictions)
            metrics["classification_report"] = classification_report(
                labels, predictions, output_dict=True, zero_division=0
            )
    except Exception as e:
        print(f"[!] Error in compute_metrics : {e}")

    return metrics


def tokenize_dataset(dataset, tokenizer, max_length=256):
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)

    tokenized_train = dataset["train"].map(tokenize_function, batched=True)
    tokenized_test = dataset["test"].map(tokenize_function, batched=True)
    tokenized_valid = dataset["validation"].map(tokenize_function, batched=True)

    return tokenized_train, tokenized_test, tokenized_valid


def train_model(tokenized_train, tokenized_test, num_labels, output_dir):
    model = RobertaForSequenceClassification.from_pretrained("FacebookAI/roberta-large", num_labels=num_labels)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=200,
        eval_steps=200,
        save_total_limit=4,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=4,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=100,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_results = trainer.evaluate()

    return trainer, eval_results


def detailed_evaluation(trainer, tokenized_valid):
    unique_tasks = set(tokenized_valid["task_id"])
    unique_levels = set(tokenized_valid["level_title"])

    def safe_metrics(ref_labels, predicted_labels):
        result = {
            "accuracy": float('nan'),
            "precision": float('nan'),
            "recall": float('nan'),
            "f1": float('nan'),
            "ck": float('nan'),
            "pearson": float('nan'),
        }

        if len(ref_labels) < 2 or len(set(ref_labels)) <= 1 or len(set(predicted_labels)) <= 1:
            return result 

        try:
            result["ck"] = round(cohen_kappa_score(predicted_labels, ref_labels, weights="quadratic"), 2)
        except Exception as e:
            print(f"[!] Error Cohen Kappa : {e}")

        try:
            result["pearson"], _ = pearsonr(ref_labels, predicted_labels)
        except Exception as e:
            print(f"[!] Error Pearson : {e}")

        try:
            result["accuracy"] = accuracy_score(ref_labels, predicted_labels)
            result["precision"], result["recall"], result["f1"], _ = precision_recall_fscore_support(
                ref_labels, predicted_labels, average="weighted", zero_division=0
            )
        except Exception as e:
            print(f"[!] Error other metrics : {e}")

        return result

    results_tasks = []
    for t in unique_tasks:
        sub_ds = tokenized_valid.filter(lambda example: example['task_id'] == t)
        if len(sub_ds) < 2:
            print(f"[!] Task {t} ignored : not enough samples")
            continue

        predictions = trainer.predict(sub_ds)
        outputs = predictions.predictions
        predicted_labels = np.argmax(outputs, axis=-1)
        ref_labels = predictions.label_ids

        metrics = safe_metrics(ref_labels, predicted_labels)

        results_tasks.append({
            "task_id": t,
            "level_title": sub_ds["level_title"][0],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "ck": metrics["ck"],
            "pearson": metrics["pearson"],
            "n_samples": len(sub_ds),
        })

    results_levels = []
    for l in unique_levels:
        sub_ds = tokenized_valid.filter(lambda example: example['level_title'] == l)
        if len(sub_ds) < 2:
            print(f"[!] Level {l} ignored : not enouth samples")
            continue

        predictions = trainer.predict(sub_ds)
        outputs = predictions.predictions
        predicted_labels = np.argmax(outputs, axis=-1)
        ref_labels = predictions.label_ids

        metrics = safe_metrics(ref_labels, predicted_labels)

        results_levels.append({
            "level_title": l,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "ck": metrics["ck"],
            "pearson": metrics["pearson"],
            "n_samples": len(sub_ds),
        })

    # Sauvegarde des résultats
    pd.DataFrame(results_tasks).to_csv("result_eval_data_roberta_large_writing_task_acc.csv", index=False)
    pd.DataFrame(results_levels).to_csv("result_eval_data_roberta_large_acc_by_level.csv", index=False)


def main():
    print("Preparing data...")
    dataset = prepare_data()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-large")

    print("Tokenizing datasets...")
    tokenized_train, tokenized_test, tokenized_valid = tokenize_dataset(dataset, tokenizer)

    unique_labels = set(dataset['train']['label'])
    num_labels = len(unique_labels)
    print(f"Number of labels: {num_labels}")

    print("Training model...")
    output_dir = "../../../model_saved/roberta-large-ft-acc-writing-task-augmented"
    trainer, eval_results = train_model(tokenized_train, tokenized_test, num_labels, output_dir)

    print("Evaluation results on test set:")
    print(eval_results)

    print("Performing detailed evaluation on validation set...")
    detailed_evaluation(trainer, tokenized_valid)


if __name__ == "__main__":
    main()
