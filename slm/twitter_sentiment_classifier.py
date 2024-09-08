#!/usr/bin/env python
# coding: utf-8

# ## Twitter Sentiment Analysis Classifier
# nohup python -u /home/ec2-user/tmp/llm_fine_tuning/slm/twitter_sentiment_classifier.py > /home/ec2-user/tmp/llm_fine_tuning/slm/nohup.out &
# tail -f /home/ec2-user/tmp/llm_fine_tuning/slm/nohup.out
# ps -ef | grep python


# ### Setup

# In[28]:


#!pip install evaluate transformers[torch] wandb


# In[29]:


import wandb

# Initialize a W&B run
wandb.init(project='twitter_sentiment_classifier', entity='knebhi')


# In[30]:


# load data from s3 bucket
import boto3
import sagemaker

region_name = 'eu-central-1'

session = boto3.Session(region_name=region_name)
s3_sess = session.client('s3')
sm_session = sagemaker.Session(boto_session=session)


# ### Load Dataset

# In[31]:


import pandas as pd
from io import StringIO


# In[32]:


# Define the bucket name and the CSV file key (path in the bucket)
bucket_name = 'sagemaker-eu-central-1-505049265445'
csv_file_key = 'datasets/twitter_ds/twitter_dataset_full.csv'

# Fetch the CSV file content from S3
obj = s3_sess.get_object(Bucket=bucket_name, Key=csv_file_key)
data = obj['Body'].read().decode('utf-8')

# Convert the content to a pandas DataFrame
df = pd.read_csv(StringIO(data))

# Display the DataFrame
print(df.head())
print(df.shape)


# In[33]:


import re
# Function to check if a tweet has at least three tokens
def has_at_least_three_tokens(message):
    tokens = message.split()
    return len(tokens) >= 3

# Function to check if a tweet contains a URL
def contains_url(message):
    # Regex to detect URLs
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return bool(re.search(url_pattern, message))

# Filter tweets with at least three tokens and no URLs
df_filtered = df[df['message'].apply(lambda x: has_at_least_three_tokens(x) and not contains_url(x))]


# In[34]:


df_filtered.shape


# In[35]:


df_filtered = df_filtered[["message","is_positive"]]
df_filtered = df_filtered.rename(columns={'is_positive': 'label', 'message': 'text'})


# In[36]:


df_sample = df_filtered.sample(n=1498948, random_state=42)  # 'random_state' ensures reproducibility
#df_sample = df  # 'random_state' ensures reproducibility


# In[37]:


df_sample["label"].value_counts()


# In[38]:


df_sample.shape


# In[39]:


from datasets import Dataset, DatasetDict

# Drop the index from the DataFrame before conversion
df_sample.reset_index(drop=True, inplace=True)

# Convert the pandas DataFrame to a Hugging Face Dataset
dataset = Dataset.from_pandas(df_sample)

# Proceed with the train/test/validation split as before
train_testvalid = dataset.train_test_split(test_size=0.2, seed=42)
test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=42)

# Combine train, validation, and test sets into a DatasetDict
dataset = DatasetDict({
    'train': train_testvalid['train'],
    'validation': test_valid['train'],
    'test': test_valid['test']
})

# Display the DatasetDict structure
print(dataset)


# In[40]:


# save train_dataset to s3 using our SageMaker session
input_path = f's3://{sm_session.default_bucket()}/datasets/twitter_ds/'

# save datasets to s3
dataset["train"].to_json(f"{input_path}/train/dataset.json", orient="records")
train_dataset_s3_path = f"{input_path}/train/dataset.json"
dataset["test"].to_json(f"{input_path}/test/dataset.json", orient="records")
test_dataset_s3_path = f"{input_path}/test/dataset.json"
dataset["validation"].to_json(f"{input_path}/validation/dataset.json", orient="records")
validation_dataset_s3_path = f"{input_path}/validation/dataset.json"


print(f"Training data uploaded to:")
print(train_dataset_s3_path)
print(test_dataset_s3_path)
print(validation_dataset_s3_path)

print(f"https://s3.console.aws.amazon.com/s3/buckets/{sm_session.default_bucket()}/?region={sm_session.boto_region_name}&prefix={input_path.split('/', 3)[-1]}/")


# ### Preprocessing

# In[41]:


model_id = "roberta-base"


# In[42]:


from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)


# In[43]:


tokenizer(dataset["train"][0]["text"])


# In[44]:


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )


# In[45]:


tokenized_train = dataset["train"].map(tokenize_function, batched=True)
tokenized_test = dataset["test"].map(tokenize_function, batched=True)
tokenized_validation = dataset["validation"].map(tokenize_function, batched=True)


# In[46]:


unique_labels = set(dataset['test']['label'])
num_labels = len(unique_labels)
num_labels


# ### Model 

# In[47]:


from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=num_labels)


# In[48]:


import numpy as np 
import evaluate

# Load the evaluation metrics
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Compute accuracy
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    
    # Compute F1 score (use 'macro', 'micro', or 'weighted' depending on your task)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    
    # Return both metrics
    return {
        'accuracy': accuracy['accuracy'],  # Access the 'accuracy' key from the accuracy result
        'f1': f1['f1'],                    # Access the 'f1' key from the F1 result
    }


# In[49]:


args = TrainingArguments(
    output_dir="../../model_saved/roberta-twitter-sa",
    eval_strategy= "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=1,
    report_to='wandb' 
)


# In[50]:


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
)


# In[51]:


trainer.train()


# In[52]:


trainer.evaluate()


# In[53]:


# Perform evaluation on the validation dataset
test_results = trainer.predict(test_dataset=tokenized_validation)


# In[54]:


# Extract predictions, labels, and metrics
predictions = test_results.predictions
labels = test_results.label_ids
metrics = test_results.metrics
metrics

