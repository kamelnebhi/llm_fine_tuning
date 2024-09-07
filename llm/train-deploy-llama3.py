#!/usr/bin/env python
# coding: utf-8

# nohup python -u /home/ec2-user/tmp/llm_fine_tuning/llm/train-deploy-llama3.py > /home/ec2-user/tmp/llm_fine_tuning/llm/nohup.out &
# tail -f /home/ec2-user/tmp/llm_fine_tuning/llm/nohup.out
# ps -ef | grep python


# # Fine-tune Llama 3 with PyTorch FSDP and Q-Lora on Amazon SageMaker
# 
# This blog post walks you thorugh how to fine-tune a Llama 3 using PyTorch FSDP and Q-Lora with the help of Hugging Face [TRL](https://huggingface.co/docs/trl/index), [Transformers](https://huggingface.co/docs/transformers/index), [peft](https://huggingface.co/docs/peft/index) & [datasets](https://huggingface.co/docs/datasets/index) on Amazon SageMAker. In addition to FSDP we will use [Flash Attention v2](https://github.com/Dao-AILab/flash-attention) implementation. 
# 
# This blog is an extension and dedicated version to my [Efficiently fine-tune Llama 3 with PyTorch FSDP and Q-Lora](https://www.philschmid.de/fsdp-qlora-llama3) version, specifically tailored to run on Amazon SageMaker.
# 
# 1. [Setup development environment](#2-setup-development-environment)
# 2. [Create and prepare the dataset](#3-create-and-prepare-the-dataset)
# 3. [Fine-tune Llama 3 on Amazon SageMaker](#4-fine-tune-llm-using-trl-and-the-sfttrainer)
# 4. [Deploy & Test fine-tuned Llama 3 on Amazon SageMaker](#5-test-and-evaluate-the-llm)
# 
# _Note: This blog was created and validated on `ml.p4d.24xlarge` and `ml.g5.48.xlarge` instances. The configurations and code are optimized for `ml.p4d.24xlarge` with 8xA100 GPUs each with 40GB of Memory. We tried `ml.g5.12xlarge` but Amazon SageMaker reserves more memory than EC2. We plan to add support for `trn1` in the coming weeks._
# 
# **FSDP + Q-Lora Background**
# 
# In a collaboration between [Answer.AI](https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html), Tim Dettmers [Q-Lora creator](https://github.com/TimDettmers/bitsandbytes) and [Hugging Face](https://huggingface.co/), we are proud to announce to share the support of Q-Lora and PyTorch FSDP (Fully Sharded Data Parallel). FSDP and Q-Lora allows you now to fine-tune Llama 2 70b or Mixtral 8x7B on 2x consumer GPUs (24GB). If you want to learn more about the background of this collaboration take a look at [You can now train a 70b language model at home](https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html). Hugging Face PEFT is were the magic happens for this happens, read more about it in the [PEFT documentation](https://huggingface.co/docs/peft/v0.10.0/en/accelerate/fsdp).
# 
# * [PyTorch FSDP](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/) is a data/model parallelism technique that shards model across GPUs, reducing memory requirements and enabling the training of larger models more efficiently​​​​​​.
# * Q-LoRA is a fine-tuning method that leverages quantization and Low-Rank Adapters to efficiently reduced computational requirements and memory footprint. 
# 
# 
# 
# 
# ## 1. Setup Development Environment
# 
# Our first step is to install Hugging Face Libraries we need on the client to correctly prepare our dataset and start our training/evaluations jobs. 

# In[1]:


#!pip install transformers datasets "sagemaker>=2.229.0" "huggingface_hub[cli]" --upgrade --quiet


# Next we need to login into Hugging Face to access the Llama 3 70b model and store our trained model on Hugging Face. If you don't have an account yet and accepted the terms, you can create one [here](https://huggingface.co/join). 

# If you are going to use Sagemaker in a local environment. You need access to an IAM Role with the required permissions for Sagemaker. You can find [here](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html) more about it.
# 
# 

# In[5]:


# load data from s3 bucket
import boto3
import sagemaker

region_name = 'eu-central-1'

session = boto3.Session(region_name=region_name)
s3_sess = session.client('s3')
sm_session = sagemaker.Session(boto_session=session)
role = "arn:aws:iam::505049265445:role/service-role/AmazonSageMaker-ExecutionRole-20220919T094604"


# ## 2. Create and prepare the dataset
# 
# 
# After our environment is set up, we can start creating and preparing our dataset. A fine-tuning dataset should have a diverse set of demonstrations of the task you want to solve. If you want to learn more about how to create a dataset, take a look at the [How to Fine-Tune LLMs in 2024 with Hugging Face](https://www.philschmid.de/fine-tune-llms-in-2024-with-trl#3-create-and-prepare-the-dataset).

# In[7]:


training_input_path = f's3://{sm_session.default_bucket()}/datasets/writing_accuracy_dataset/train_dataset_writing_accuracy.json'
# define a data input dictonary with our uploaded s3 uris
data = {'training': training_input_path}
data


# In[8]:


from datasets import load_dataset, DatasetDict
raw_dataset = load_dataset(
        "json",
        data_files=data
    )
raw_dataset


# In[9]:


indices_1 = range(0,7000)
indices_2 = range(7001,7527)
dataset_dict = {
    "train": raw_dataset["training"].select(indices_1),
    "test": raw_dataset["training"].select(indices_2)
}
dataset = DatasetDict(dataset_dict)
dataset


# In[12]:


dataset["train"][14]


# After we processed the datasets we are going to use the [FileSystem integration](https://huggingface.co/docs/datasets/filesystems) to upload our dataset to S3. We are using the `sess.default_bucket()`, adjust this if you want to store the dataset in a different S3 bucket. We will use the S3 path later in our training script.

# In[13]:


# save train_dataset to s3 using our SageMaker session
input_path = f's3://{sm_session.default_bucket()}/datasets/llama3-writing_acc-instruct'

# save datasets to s3
dataset["train"].to_json(f"{input_path}/train/dataset.json", orient="records")
train_dataset_s3_path = f"{input_path}/train/dataset.json"
dataset["test"].to_json(f"{input_path}/test/dataset.json", orient="records")
test_dataset_s3_path = f"{input_path}/test/dataset.json"

print(f"Training data uploaded to:")
print(train_dataset_s3_path)
print(test_dataset_s3_path)
print(f"https://s3.console.aws.amazon.com/s3/buckets/{sm_session.default_bucket()}/?region={sm_session.boto_region_name}&prefix={input_path.split('/', 3)[-1]}/")


# ## 3. Fine-tune Llama 3 on Amazon SageMaker
# 
# We are now ready to fine-tune our model. We will use the [SFTTrainer](https://huggingface.co/docs/trl/sft_trainer) from `trl` to fine-tune our model. The `SFTTrainer` makes it straightfoward to supervise fine-tune open LLMs. The `SFTTrainer` is a subclass of the `Trainer` from the `transformers`. We prepared a script [run_fsdp_qlora.py](../scripts/fsdp/run_fsdp_qlora.py) which will loads the dataset from disk, prepare the model, tokenizer and start the training. It usees the [SFTTrainer](https://huggingface.co/docs/trl/sft_trainer) from `trl` to fine-tune our model. The `SFTTrainer` makes it straightfoward to supervise fine-tune open LLMs supporting:
# * Dataset formatting, including conversational and instruction format (✅ used)
# * Training on completions only, ignoring prompts (❌ not used)
# * Packing datasets for more efficient training (✅ used)
# * PEFT (parameter-efficient fine-tuning) support including Q-LoRA (✅ used)
# * Preparing the model and tokenizer for conversational fine-tuning (❌ not used, see below)
# 
# For configuration we use the new `TrlParser`, that allows us to provide hyperparameters in a yaml file. This `yaml` will be uploaded and provided to Amazon SageMaker similar to our datasets. Below is the config file for fine-tuning Llama 3 70B on 8x A100 GPUs or 4x24GB GPUs. We are saving the config file as `fsdp_qlora_llama3_70b.yaml` and upload it to S3.


# In[25]:

from sagemaker.s3 import S3Uploader

# upload the model yaml file to s3
model_yaml = "llama_3_8b_instruct_fsdp_qlora.yaml"
train_config_s3_path = S3Uploader.upload(local_path=model_yaml, desired_s3_uri=f"{input_path}/config")

print(f"Training config uploaded to:")
print(train_config_s3_path)


# In order to create a sagemaker training job we need an `HuggingFace` Estimator. The Estimator handles end-to-end Amazon SageMaker training and deployment tasks. The Estimator manages the infrastructure use. Amazon SagMaker takes care of starting and managing all the required ec2 instances for us, provides the correct huggingface container, uploads the provided scripts and downloads the data from our S3 bucket into the container at `/opt/ml/input/data`. Then, it starts the training job by running.
# 
# > Note: Make sure that you include the `requirements.txt` in the `source_dir` if you are using a custom training script. We recommend to just clone the whole repository.
# 
# To use `torchrun` to execute our scripts, we only have to define the `distribution` parameter in our Estimator and set it to `{"torch_distributed": {"enabled": True}}`. This tells SageMaker to launch our training job with.
# 
# ```python
# torchrun --nnodes 2 --nproc_per_node 8 --master_addr algo-1 --master_port 7777 --node_rank 1 run_fsdp_qlora.py --config /opt/ml/input/data/config/config.yaml
# ```
# 
# The `HuggingFace` configuration below will start a training job on 1x `p4d.24xlarge` using 8x A100 GPUs. The amazing part about SageMaker is that you can easily scale up to 2x `p4d.24xlarge` by modifying the `instance_count`. SageMaker will take care of the rest for you. 

# In[21]:


from sagemaker.huggingface import HuggingFace
from huggingface_hub import HfFolder

# define Training Job Name 
job_name = f'llama3-instruct-8b-writing-acc-exp1'

# create the Estimator
huggingface_estimator = HuggingFace(
    entry_point          = 'run_fsdp_qlora.py',      # train script
    source_dir           = './scripts/fsdp',  # directory which includes all the files needed for training
    instance_type        = 'ml.g5.12xlarge',  # instances type used for the training job
    instance_count       = 1,                 # the number of instances used for training
    max_run              = 2*24*60*60,        # maximum runtime in seconds (days * hours * minutes * seconds)
    base_job_name        = job_name,          # the name of the training job
    role                 = role,              # Iam role used in training job to access AWS ressources, e.g. S3
    volume_size          = 200,               # the size of the EBS volume in GB
    transformers_version = '4.36.0',          # the transformers version used in the training job
    pytorch_version      = '2.1.0',           # the pytorch_version version used in the training job
    py_version           = 'py310',           # the python version used in the training job
    hyperparameters      =  {
        "config": "/opt/ml/input/data/config/llama_3_8b_instruct_fsdp_qlora.yaml" # path to TRL config which was uploaded to s3
    },
    disable_output_compression = True,        # not compress output to save training time and cost
    distribution={"torch_distributed": {"enabled": True}},   # enables torchrun
    environment  = {
        "HUGGINGFACE_HUB_CACHE": "/tmp/.cache", # set env variable to cache models in /tmp
        "HF_TOKEN": HfFolder.get_token(),       # huggingface token to access gated models, e.g. llama 3
        "ACCELERATE_USE_FSDP": "1",             # enable FSDP
        "FSDP_CPU_RAM_EFFICIENT_LOADING": "1"   # enable CPU RAM efficient loading
    }, 
)


# _Note: When using QLoRA, we only train adapters and not the full model. The [run_fsdp_qlora.py](../scripts/fsdp/run_fsdp_qlora.py) merges the `base_model`with the `adapter` at the end of the training to directly be able to deploy to Amazon SageMaker._
# 
# We can now start our training job, with the `.fit()` method passing our S3 path to the training script.

# In[22]:


# define a data input dictonary with our uploaded s3 uris
data = {
  'train': train_dataset_s3_path,
  'test': test_dataset_s3_path,
  'config': train_config_s3_path
  }

# starting the train job with our uploaded datasets as input
huggingface_estimator.fit(data, wait=True)
