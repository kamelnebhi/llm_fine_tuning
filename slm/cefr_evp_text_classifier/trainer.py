#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
training_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
training_device
import numpy as np


# In[ ]:


import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict

# Define the mapping from labels to values
label_mapping = {
    'A1': 0,
    'A2': 1,
    'B1': 2,
    'B2': 3,
    'C1': 4,
    'C2': 5
}

df = pd.read_csv("data/evp.csv")

# Apply the mapping to the 'labels' column
df['label'] = df['label'].map(label_mapping)
df.dropna(subset=['label'], inplace=True)
df.reset_index(drop=True, inplace=True)

df.head()


# In[ ]:


df["label"].value_counts()


# In[ ]:


ds = Dataset.from_pandas(df)
ds


# In[ ]:


from datasets import ClassLabel, Value, Sequence
new_features = ds.features.copy()
new_features["label"] = ClassLabel(names=[0, 1, 2, 3, 4, 5])
ds = ds.cast(new_features)

# Step 1: Initial train/test split with stratification
train_test_ds = ds.train_test_split(test_size=0.20, seed=20)

# Step 2: Split the test set into half test, half validation
test_valid_split = train_test_ds['test'].train_test_split(test_size=0.5, seed=20)

# Step 3: Combine everything into a single DatasetDict
ds = DatasetDict({
    'train': train_test_ds['train'],
    'test': test_valid_split['train'],    # This becomes the test set
    'validation': test_valid_split['test']  # This becomes the validation set
})
ds


# In[ ]:


# Verify label distribution
from collections import Counter

print("Train label counts:", Counter(ds['train']['label']))
print("Test label counts:", Counter(ds['test']['label']))
print("Validation label counts:", Counter(ds['validation']['label']))


# In[ ]:


from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


# In[ ]:


tokenizer(ds["train"][0]["text"])


# In[ ]:


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )


# In[ ]:


tokenized_train = ds["train"].map(tokenize_function, batched=True)


# In[ ]:


tokenized_test = ds["test"].map(tokenize_function, batched=True)


# In[ ]:


tokenized_validation = ds["validation"].map(tokenize_function, batched=True)


# In[ ]:


unique_labels = set(ds['train']['label'])
num_labels = len(unique_labels)
num_labels


# In[ ]:


from transformers import DistilBertForSequenceClassification, TrainingArguments, Trainer
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)


# In[ ]:


import numpy as np 
import evaluate

metric = evaluate.load("accuracy")


# In[ ]:


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# In[ ]:


args = TrainingArguments(
    output_dir="../../model_saved/distilbert-large-ft-speaking-range",
    evaluation_strategy= "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)


# In[ ]:


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
)


# In[ ]:


trainer.train()


# In[ ]:


print(trainer.evaluate())


# In[ ]:


predictions = trainer.predict(tokenized_test)
logits = predictions.predictions

predic_ = np.argmax(logits, axis=-1)
ref = predictions.label_ids
#print(predic_)
#print(predictions.predictions, predictions.label_ids)

from sklearn.metrics import cohen_kappa_score
ck = round(cohen_kappa_score(predic_, ref, weights="quadratic"), 2)
print("cohen kappa==> ",ck)

from sklearn.metrics import classification_report
print(classification_report(ref, predic_))

