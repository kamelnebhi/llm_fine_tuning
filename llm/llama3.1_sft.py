#!/usr/bin/env python
# coding: utf-8

# ## Fine tune a llama3 model with m-a-p/Code-Feedback

# ### Setup
# Let's start by installing all the lib we need to do supervised fine-tuning. We're going to use
# 
# Transformers for the LLM which we're going to fine-tune
# Datasets for loading a SFT dataset from the hub, and preparing it for the model
# BitsandBytes and PEFT for fine-tuning the model on consumer hardware, leveraging Q-LoRa, a technique which drastically reduces the compute requirements for fine-tuning
# TRL, a library which includes useful Trainer classes for LLM fine-tuning.

# In[ ]:

# nohup python -u /home/ec2-user/tmp/llm_fine_tuning/llm/llama3.1_sft.py > /home/ec2-user/tmp/llm_fine_tuning/llm/nohup.out &
# tail -f /home/ec2-user/tmp/llm_fine_tuning/llm/nohup.out
# ps -ef | grep python


#!pip install -q transformers[torch] datasets
#!pip install -q bitsandbytes trl peft
#!pip install flash-attn --no-build-isolation


# In[ ]:


import torch
torch.cuda.empty_cache()


# ### Load Data + Preprocessing
# The dataset contains various splits, each with a certain number of rows. In our case, as we're going to do supervised fine-tuning (SFT), only the "train_sft" and "test_sft" splits are relevant for us.
# 

# In[ ]:


from datasets import load_dataset, DatasetDict


# In[ ]:


# load data from s3 bucket
import boto3
import sagemaker

region_name = 'eu-central-1'

session = boto3.Session(region_name=region_name)
s3_sess = session.client('s3')
sm_session = sagemaker.Session(boto_session=session)


# In[ ]:


training_input_path = f's3://{sm_session.default_bucket()}/datasets/writing_accuracy_dataset/train_dataset.json'
# define a data input dictonary with our uploaded s3 uris
data = {'training': training_input_path}

data


# In[ ]:


raw_dataset = load_dataset(
        "json",
        data_files=data
    )
raw_dataset


# In[ ]:


indices_1 = range(0,7000)
indices_2 = range(7001,7527)
dataset_dict = {
    "train": raw_dataset["training"].select(indices_1),
    "test": raw_dataset["training"].select(indices_2)
}
raw_dataset = DatasetDict(dataset_dict)
raw_dataset


# #### Tokenizer
# Next, we instantiate the tokenizer, which is required to prepare the text for the model. The model doesn't directly take strings as input, but rather input_ids, which represent integer indices in the vocabulary of a Transformer model. 
# 
# We also set some attributes which the tokenizer of a base model typically doesn't have set, such as:
# 
# - the padding token ID. During pre-training, one doesn't need to pad since one just creates blocks of text to predict the next token, but during fine-tuning, we will need to pad the (instruction, completion) pairs in order to create batches of equal length.
# - the model max length: this is required in order to truncate sequences which are too long for the model. Here we decide to train on at most 2048 tokens.
# - the chat template. A chat template determines how each list of messages is turned into a tokenizable string, by adding special strings in between such as <|user|> to indicate a user message and <|assistant|> to indicate the chatbot's response. Here we define the default chat template, used by most chat models. See also the docs.

# In[ ]:


from transformers import AutoTokenizer
from huggingface_hub import login


# In[ ]:


model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"


# In[ ]:


tokenizer = AutoTokenizer.from_pretrained(model_id)

# set pad_token_id equal to the eos_token_id if not set
if tokenizer.pad_token_id is None:
  tokenizer.pad_token_id = tokenizer.eos_token_id

# Set reasonable default for models without max length
if tokenizer.model_max_length > 100_000:
  tokenizer.model_max_length = 2048


# In[ ]:


# use default template of instruct model
tokenizer.chat_template


# #### Apply chat template
# Once we have equipped the tokenizer with the appropriate attributes, it's time to apply the chat template to each list of messages. Here we basically turn each list of (instruction, completion) messages into a tokenizable string for the model.
# 
# Note that we specify tokenize=False here, since the SFTTrainer which we'll define later on will perform the tokenization internally. Here we only turn the list of messages into strings with the same format.

# In[ ]:


import re
import random
from multiprocessing import cpu_count

def apply_chat_template(example, tokenizer):
    messages = example["messages"]
    # We add an empty system message if there is none
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": ""})
    example["text"] = tokenizer.apply_chat_template(messages, tokenize=False)

    return example


# In[ ]:


column_names = list(raw_dataset["train"].features)
column_names


# In[ ]:


# applies the apply_chat_template function to each element in raw_dataset using multiple CPU cores, passing a tokenizer and removing specified columns, with a progress description.
raw_dataset = raw_dataset.map(apply_chat_template,
                                num_proc=cpu_count(),
                                fn_kwargs={"tokenizer": tokenizer},
                                remove_columns=column_names,
                                desc="Applying chat template")


# In[ ]:


train_data = raw_dataset["train"]
test_data = raw_dataset["test"]

for index in random.sample(range(len(raw_dataset["train"])), 3):
  print(f"Sample {index} of the processed training set:\n\n{raw_dataset['train'][index]['text']}")
  


# ### Model Definition
# 
# With regular LoRa, one would keep the base model in 32 or 16 bits in memory, and then train the parameter weights. However, there have been new methods developed to shrink the size of a model considerably, to 8 or 4 bits per parameter (we call this "quantization"). Hence, if we apply LoRa to a quantized model (like a 4-bit model), then we call this QLoRa. We have a blog post that tells you all about it. There are various quantization methods available, here we're going to use the BitsandBytes integration.
# 
# 

# In[ ]:


# QloRa SFT
from transformers import BitsAndBytesConfig
import torch

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Load the model with 4-bit precision to reduce memory usage.
    bnb_4bit_quant_type="nf4",  # Use the "nf4" quantization type for 4-bit precision.
    bnb_4bit_compute_dtype=torch.bfloat16  # Perform computations using the bfloat16 data type.
)

device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None
device_map


# In[ ]:


model_kwargs = dict(
    #attnt_implementation=True,  # (Commented out) Use a specific attention implementation.
    torch_dtype="auto",  # Automatically choose the appropriate torch data type (e.g., float32, float16).
    use_cache=False,  # Disable caching to save memory since gradient checkpointing is used.
    device_map=device_map,  # Map model layers to specific devices (e.g., GPUs).
    quantization_config=quant_config.to_dict()  # Apply model quantization settings by converting them to a dictionary.
)


# ### SFT Trainer
# 
# Next, we define the SFTTrainer available in the TRL library. This class inherits from the Trainer class available in the Transformers library, but is specifically optimized for supervised fine-tuning (instruction tuning). It can be used to train out-of-the-box on one or more GPUs, using Accelerate as backend.
# 
# Most notably, it supports packing, where multiple short examples are packed in the same input sequence to increase training efficiency.
# 
# As we're going to use QLoRa, the PEFT library provides a handy LoraConfig which defines on which layers of the base model to apply the adapters. One typically applies LoRa on the linear projection matrices of the attention layers of a Transformer. We then provide this configuration to the SFTTrainer class. The weights of the base model will be loaded as we specify the model_id (this requires some time).
# 
# We also specify various hyperparameters regarding training, such as:
# 
# - we're going to fine-tune for 1 epoch
# - the learning rate and its scheduler
# - we're going to use gradient checkpointing (yet another way to save memory during training)
# - and so on.

# In[ ]:


from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from transformers import TrainingArguments


# In[ ]:


output_dir = "../../model_saved/llama3.1_accuracy_writing-gen-8B"


# In[ ]:


# based on config
training_args = TrainingArguments(
    fp16=True,  # Use 16-bit floating point precision (faster and less memory).
    do_eval=True,  # Perform evaluation during training.
    eval_strategy="epoch",  # Evaluate the model at the end of each training epoch.
    gradient_accumulation_steps=128,  # Accumulate gradients over 128 steps before updating weights.
    gradient_checkpointing=True,  # Save memory by not storing intermediate activations.
    gradient_checkpointing_kwargs={"use_reentrant": False},  # Disable reentrant autograd for gradient checkpointing.
    learning_rate=2.0e-05,  # Set the learning rate for the optimizer.
    log_level="info",  # Set the level of detail for logging information.
    logging_steps=5,  # Log information every 5 steps.
    logging_strategy="steps",  # Log based on steps, not epochs.
    lr_scheduler_type="cosine",  # Use a cosine schedule to adjust the learning rate.
    max_steps=-1,  # Train indefinitely unless a stopping criterion is met.
    num_train_epochs=4,  # Train for 1 full epoch.
    output_dir=output_dir,  # Directory to save training outputs and checkpoints.
    overwrite_output_dir=True,  # Overwrite the output directory if it exists.
    per_device_eval_batch_size=1,  # Use a batch size of 1 per device for evaluation (previously set to 8).
    per_device_train_batch_size=1,  # Use a batch size of 1 per device for training (previously set to 8).
    save_strategy="no",  # Don't save model checkpoints during training.
    save_total_limit=None,  # No limit on the number of checkpoints saved.
    seed=42,  # Set the random seed for reproducibility.
)

# based on config
peft_config = LoraConfig(
    r=64,  # Set the rank for the LoRA (low-rank adaptation) matrices.
    lora_alpha=16,  # Scaling factor for the LoRA updates.
    lora_dropout=0.1,  # Apply dropout with a probability of 0.1 to LoRA layers.
    bias="none",  # No additional bias terms are added in the LoRA layers.
    task_type="CAUSAL_LM",  # Specify the task type as causal language modeling.
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Apply LoRA to these specific model modules.
)


# In[ ]:


trainer = SFTTrainer(
    model=model_id,  # Use the specified model ID for training.
    model_init_kwargs=model_kwargs,  # Pass in model initialization arguments (like device mapping and dtype).
    args=training_args,  # Apply the defined training arguments (e.g., learning rate, batch size).
    train_dataset=train_data,  # Use the provided dataset for training.
    eval_dataset=test_data,  # Use the provided dataset for evaluation.
    dataset_text_field="text",  # Specify the field containing text data in the datasets.
    tokenizer=tokenizer,  # Use the specified tokenizer for preprocessing the text data.
    packing=True,  # Enable sequence packing to maximize training efficiency.
    peft_config=peft_config,  # Apply the specified parameter-efficient fine-tuning (PEFT) configuration.
    max_seq_length=tokenizer.model_max_length,  # Set the maximum sequence length for the model based on the tokenizer's max length.
)


# In[ ]:


train_result = trainer.train()


# ### Saving Model

# In[ ]:


metrics = train_result.metrics
max_train_samples = 100
metrics["train_samples"] = min(max_train_samples, len(train_data))
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()
trainer.save_model(training_args.output_dir)


# In[ ]:


# save to s3 bucket
# Define your S3 bucket name and key (path in the bucket)
bucket_name = 'sagemaker-eu-central-1-505049265445'
model_s3_key = 'models/llama3.1_accuracy_writing-gen-8B'  # Adjust the path and model name

# Upload all files from the output directory to the S3 bucket
import os

for root, dirs, files in os.walk(output_dir):
    for file in files:
        # Construct the full local file path
        local_file_path = os.path.join(root, file)
        
        # Construct the full S3 path
        s3_file_path = os.path.join(model_s3_key, os.path.relpath(local_file_path, output_dir))
        
        # Upload the file
        s3_sess.upload_file(local_file_path, bucket_name, s3_file_path)
        print(f"Uploaded {local_file_path} to s3://{bucket_name}/{s3_file_path}")


# ### Inference

'''
from transformers import AutoTokenizer, AutoModelForCausalLM

output_dir = "../../model_saved/llama3.1_accuracy_writing-gen-8B"

tokenizer = AutoTokenizer.from_pretrained(output_dir)
model = AutoModelForCausalLM.from_pretrained(output_dir, load_in_4bit=True, device_map="auto")


import torch

# We use the tokenizer's chat template to format each message
#messages = [
#    {   
#        "role": "user", "content": "Answer the following prompt: 'What did you do during your last vacation?' with a level of accuracy 6"
#    },
#]
messages = [{"content":"You will evaluate a student's response to a question, focusing on their grammatical accuracy in their second language. \nYour role is that of an English teacher assessing the student's ability to use grammatical structures \/ sentence patterns correctly. \nThe scoring criteria range from 1 to 6, with each level representing increasing proficiency in grammatical structures.\n\nHere is the scale you should use to build your answer:\n1: The student_answer has frequent and noticeable Errors: The student makes frequent and noticeable errors in grammar, significantly impeding intelligibility.\n2: The student_answer has basic Control with consistent errors: The student demonstrates control over simple grammatical structures but makes consistent basic errors that often impede intelligibility.\n3: The student_answer has control with some errors: The student demonstrates control over common grammatical structures, although occasional errors impede intelligibility.\n4: The student_answer has fair control with infrequent errors: The student demonstrates control over a variety of grammatical structures, with noticeable errors made infrequently. These errors do not significantly impede intelligibility, and there may be some self-correction.\n5: The student_answer has high control with rare errors: The student demonstrates a high level of control over a variety of grammatical structures, with errors rarely made. However, there might be noticeable errors that occasionally impede intelligibility, and self-correction may be inconsistent.\n6: The student_answer has advanced control with rare errors: The student demonstrates advanced control over a variety of grammatical structures, even when engaged in other activities. Errors are rarely noticeable, and there is consistent evidence of self-correction.\n","role":"system"},
            {"content":"Answer the following prompt: 'What did you do during your last vacation?' with a level of accuracy 1","role":"user"},
          ]


#DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '' }}\n{% endif %}\n{% endfor %}"
#tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

# Prepare the messages for the model
input_ids = tokenizer.apply_chat_template(messages, truncation=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

# Ensure input_ids are valid
#print("Input IDs:", input_ids)

# Inference with model output checks
outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=256,
        do_sample=True,
        temperature=1.0,  # Adjusted for stability
        top_k=2,
        top_p=0.9,
        return_dict_in_generate=True,
        output_scores=True
)

# Decode the generated output
decoded_output = tokenizer.batch_decode(outputs['sequences'], skip_special_tokens=True)[0]
print(decoded_output)
'''
