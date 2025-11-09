import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from datasets import load_dataset,Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from qwen_vl_utils import process_vision_info
from trl import GRPOConfig, GRPOTrainer
from transformers import (AutoModelForCausalLM, AutoTokenizer, Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig)
import torch
import deepspeed
DS_CONFIG = "ds_z2_offload_config.json"
import json
from dataset import get_med_dataset,get_mask_dataset,get_coco_dataset
from PIL import Image
import base64
from io import BytesIO
import pandas as pd
from tqdm import tqdm
from trl import GRPOConfig
from grpo_trainer import Qwen2VLGRPOTrainer # third-party trainer from open-R1
import re
from reward_zoo import format_reward_func,levenshtein_reward_func, reward_CoMa_v1_func




################arg########################
sft_path = '/sshfs/pretrains/Qwen/Qwen2.5-VL-3B-Instruct/'
model_path = '/sshfs/pretrains/Qwen/Qwen2.5-VL-3B-Instruct/'
output_dir="./outputs/Qwevl-Instruct-rl-0713"
#output_dir="./outputs/Qwevl-Instruct-RL"

run_name="Qwen-vl-GRPO"
###########################################


compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
)

device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
tokenizer = AutoProcessor.from_pretrained(model_path,use_fast=True)
min_pixels = 256*28*28
max_pixels = 1280*28*28
processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels,use_fast=True)
# use cuda device
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(sft_path,
                                             # device_map=device_map,
                                            torch_dtype=compute_dtype,
                                            #quantization_config=bnb_config #量化参数
                                                          )

#prepare dataset
# dataset_train = get_med_dataset()
# dataset_train = get_mask_dataset()
dataset_train = get_coco_dataset()


model.train()
peft_config = LoraConfig(
    r=4, #Rank
    lora_alpha=32,
    target_modules=[
        "q_proj", 
        "k_proj", 
        "v_proj", 
        "o_proj", 
        # "gate_proj", 
        # "up_proj", 
        # "down_proj"
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
)
 
training_args = GRPOConfig(
    # use_vllm = True, # use vLLM for fast inference!
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "adamw_torch",
    # optim = "",
    logging_steps = 1,
    bf16 = True,
    fp16 = False,
    per_device_train_batch_size = 1,# keep same with num_generations
    gradient_accumulation_steps = 2, # Increase to 4 for smoother training
    num_generations = 4, # Decrease if out of memory
    max_prompt_length = 2048,
    max_completion_length = 256,
    num_train_epochs = 4, # Set to 1 for a full training run
    # max_steps = 100,
    save_steps = 100,

    max_grad_norm = 0.1,
    report_to = "none", # Can use Weights & Biases
    output_dir = "outputs",
    deepspeed=DS_CONFIG,
    disable_tqdm=False,
)
trainer = Qwen2VLGRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        format_reward_func, # all reward functions
        reward_CoMa_v1_func],
    args=training_args,
    train_dataset=dataset_train,
    # peft_config = peft_config,
)
 
trainer.train()
 
trainer.save_model(output_dir)