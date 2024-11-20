#!/usr/bin/env python
# coding: utf-8

# ### Create DS

# In[ ]:


import torch
training_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.cuda.empty_cache()


# In[ ]:


import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict

df = pd.read_csv("data/accuracy.csv")
df = df[["prompt", "transcription", "most_common_value"]]

df = df.rename(columns={'transcription': 'text', 'most_common_value': 'label'})
df.head()


# In[ ]:


df['label'].value_counts()
ds = Dataset.from_pandas(df)
ds


# In[ ]:


from datasets import ClassLabel, Value, Sequence
new_features = ds.features.copy()
new_features["label"] = ClassLabel(names=[0, 1, 2, 3, 4])
ds = ds.cast(new_features)

# Step 1: Initial train/test split with stratification
train_test_ds = ds.train_test_split(test_size=0.30, seed=20, stratify_by_column='label')

# Step 2: Split the test set into half test, half validation
test_valid_split = train_test_ds['test'].train_test_split(test_size=0.5, seed=20, stratify_by_column='label')

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


# ### SetFit examples

# In[ ]:


from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset

# Simulate the few-shot regime by sampling 8 examples per class
train_dataset = sample_dataset(ds["train"], label_column="label", num_samples=50)
eval_dataset = ds["validation"].select(range(100))
test_dataset = ds["test"].select(range(100))


# In[ ]:


# Load a SetFit model from Hub
model = SetFitModel.from_pretrained(
    "sentence-transformers/paraphrase-mpnet-base-v2",
    labels=[0,1,2,3,4],
)

args = TrainingArguments(
    batch_size=64,
    num_epochs=4,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    metric="accuracy",
    column_mapping={"text": "text", "label": "label"}  # Map dataset columns to text/label expected by trainer
)


# In[ ]:


# Train and evaluate
trainer.train()


# In[ ]:


metrics = trainer.evaluate(test_dataset)
print(metrics)
# {'accuracy': 0.8691709844559585}


# In[ ]:


# Run inference
#preds =  model.predict(["i loved the spiderman movie!", "pineapple on pizza is the worst 🤮"])
#print(preds)
# ["positive", "negative"]


# In[ ]:


#model.save_pretrained("setfit-bge-small-v1.5-sst2-8-shot")

