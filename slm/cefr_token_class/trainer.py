#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datasets import load_dataset
from transformers import BertTokenizerFast
import numpy as np

# Initialize tokenizer (e.g., BERT tokenizer)
tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")


dataset = load_dataset("Alex123321/english_cefr_dataset")
dataset


# In[ ]:


# Split into train and temporary dataset (for validation and test)
train_ds, temp_ds = dataset['train'].train_test_split(test_size=0.2).values()

# Split the temporary dataset into validation and test
val_ds, test_ds = temp_ds.train_test_split(test_size=0.5).values()

# Verify the splits
print(f"Train size: {len(train_ds)}")
print(f"Validation size: {len(val_ds)}")
print(f"Test size: {len(test_ds)}")

# Create a DatasetDict to organize the splits
split_dataset = {
    'train': train_ds,
    'validation': val_ds,
    'test': test_ds
}

# Alternatively, if you want to have it in a DatasetDict format:
from datasets import DatasetDict
dataset_split = DatasetDict(split_dataset)


# In[ ]:


dataset_split["train"][0]


# In[ ]:


unique_labels = set(dataset['train']['ud_word_level'])  # Assuming you have the full dataset
label2id = {label: idx for idx, label in enumerate(sorted(unique_labels))}
label2id


# In[ ]:


def preprocess_function(examples):
    # Tokenize the word (this will split compound words into subwords if necessary)
    tokenized_inputs = tokenizer(examples['ud_word'], padding=True, truncation=True, is_split_into_words=False)

    # Convert labels to numeric IDs for the ud_word_level
    word_ids = tokenized_inputs.word_ids()  # This gives the mapping from token to word ID
    
    # Assign the same label to all tokens belonging to the same word
    labels = []
    for word_id in word_ids:
        if word_id is None:
            labels.append(-100)  # Padding tokens should be ignored, hence the label -100
        else:
            # Assign the label for the entire word (based on the ud_word_level)
            labels.append(label2id[examples['ud_word_level']])  # Use the level label for the word

    # Add the labels to the tokenized inputs
    tokenized_inputs['labels'] = labels
    return tokenized_inputs


# In[ ]:


# Apply preprocessing to each split of the dataset (train, validation, and test)
train_ds = dataset_split['train'].map(preprocess_function, batched=False)
val_ds = dataset_split['validation'].map(preprocess_function, batched=False)
test_ds = dataset_split['test'].map(preprocess_function, batched=False)

# Check the first processed item
print(train_ds[0])


# In[ ]:


from transformers import DataCollatorForTokenClassification

# Initialize the data collator
data_collator = DataCollatorForTokenClassification(tokenizer)


# In[ ]:


from transformers import BertForTokenClassification, Trainer, TrainingArguments

# Initialize the model (BERT for token classification)
model = BertForTokenClassification.from_pretrained("bert-base-cased", num_labels=len(label2id))

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",  # Output directory for model checkpoints
    evaluation_strategy="epoch",  # Evaluate after each epoch
    save_strategy="epoch",
    learning_rate=2e-5,  # Learning rate
    per_device_train_batch_size=32,  # Batch size for training
    per_device_eval_batch_size=32,  # Batch size for evaluation
    num_train_epochs=5,  # Number of training epochs
    weight_decay=0.01,  # Weight decay
    save_total_limit=1,               # keep only the best model (deletes older checkpoints)
    load_best_model_at_end=True,      # load the best model after training
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator,
    tokenizer=tokenizer
)

# Start training
trainer.train()


# In[ ]:


eval = trainer.evaluate()
print(eval)


# ### load model 

# In[93]:


'''
from transformers import AutoModelForTokenClassification
import torch 
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("results/checkpoint-950")
model = AutoModelForTokenClassification.from_pretrained("results/checkpoint-950")
with torch.no_grad():
    logits = model(**inputs).logits
# Example input text for token classification
text = "All you need is guigui"

# Tokenize the text using the loaded tokenizer
inputs = tokenizer(text, return_tensors="pt")

# Get the model's predictions (logits) from the tokenized input
outputs = model(**inputs)

# The logits are typically in shape (batch_size, sequence_length, num_labels)
logits = outputs.logits

# Get the predicted labels by taking the argmax of the logits
predictions = logits.argmax(dim=-1)

# Convert the predictions to label indices (if needed)
predicted_labels = predictions[0].tolist()

print(predicted_labels)
'''

