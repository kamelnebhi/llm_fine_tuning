#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
training_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
training_device


# In[2]:


import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict

df = pd.read_csv("../setfit/data/accuracy.csv")
df = df[["prompt", "transcription", "most_common_value"]]

df = df.rename(columns={'transcription': 'text', 'most_common_value': 'label'})
df.head()


# In[25]:


df['label'].value_counts()
ds = Dataset.from_pandas(df)
ds


# In[26]:


from datasets import ClassLabel, Value, Sequence
new_features = ds.features.copy()
new_features["label"] = ClassLabel(names=[0, 1, 2, 3, 4])
ds = ds.cast(new_features)

# Step 1: Initial train/test split with stratification
train_test_ds = ds.train_test_split(test_size=0.20, seed=20, stratify_by_column='label')

# Step 2: Split the test set into half test, half validation
test_valid_split = train_test_ds['test'].train_test_split(test_size=0.5, seed=20, stratify_by_column='label')

# Step 3: Combine everything into a single DatasetDict
ds = DatasetDict({
    'train': train_test_ds['train'],
    'test': test_valid_split['train'],    # This becomes the test set
    'validation': test_valid_split['test']  # This becomes the validation set
})
ds


# In[27]:


# Verify label distribution
from collections import Counter

print("Train label counts:", Counter(ds['train']['label']))
print("Test label counts:", Counter(ds['test']['label']))
print("Validation label counts:", Counter(ds['validation']['label']))


# ## Train

# In[28]:


from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")


# In[8]:


tokenizer(ds["train"][0]["text"])


# In[29]:


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )


# In[30]:


tokenized_train = ds["train"].map(tokenize_function, batched=True)


# In[31]:


tokenized_test = ds["test"].map(tokenize_function, batched=True)


# In[32]:


unique_labels = set(ds['train']['label'])
num_labels = len(unique_labels)
num_labels


# In[33]:


from transformers import BertForSequenceClassification, TrainingArguments, Trainer
model = BertForSequenceClassification.from_pretrained("bert-large-uncased", num_labels=num_labels)


# In[34]:


import numpy as np 
import evaluate

metric = evaluate.load("accuracy")


# In[35]:


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# In[36]:


args = TrainingArguments(
    output_dir="../../model_saved/bert-large-ft--speaking-acc",
    evaluation_strategy= "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)


# In[37]:


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
)


# In[38]:


trainer.train()


# In[21]:


eval = trainer.evaluate()
print(eval)


# In[ ]:




