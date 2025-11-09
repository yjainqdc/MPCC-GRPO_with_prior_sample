import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import torch
from datasets import Dataset
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer
)
import json
from image_utils import *
from accelerate import Accelerator
from dataset import get_med_dataset,get_mask_dataset,get_coco_dataset,get_coco_sft_dataset



accelerator = Accelerator()

def process_func(example):

    MAX_LENGTH = 8192
    input_ids, attention_mask, labels = [], [], []
    conversation = example["messages"]
    input_content = conversation[0]["content"]
    output_content = conversation[1]["content"]
    # file_path = input_content.split("<|vision_start|>")[1].split("<|vision_end|>")[0]

    file_path = example['image']
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"{file_path}",
                    "resized_height": 280,
                    "resized_width": 280,
                },
                {"type": "text", "text": f"{input_content}"},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )  # 获取文本
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.tolist() for key, value in inputs.items()}
    instruction = inputs

    response = tokenizer(f"{output_content}", add_special_tokens=False)

    input_ids = (
            instruction["input_ids"][0] + response["input_ids"] + [tokenizer.pad_token_id]
    )

    attention_mask = instruction["attention_mask"][0] + response["attention_mask"] + [1]
    labels = (
            [-100] * len(instruction["input_ids"][0])
            + response["input_ids"]
            + [tokenizer.pad_token_id]
    )
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    labels = torch.tensor(labels)
    inputs['pixel_values'] = torch.tensor(inputs['pixel_values'])
    inputs['image_grid_thw'] = torch.tensor(inputs['image_grid_thw']).squeeze(0)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels,
            "pixel_values": inputs['pixel_values'], "image_grid_thw": inputs['image_grid_thw']}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='argparse')
    parser.add_argument("--pretrained_model", type=str, default="/sshfs/pretrains/Qwen/Qwen2.5-VL-3B-Instruct/", help="Pretrained model to use.")
    parser.add_argument("--batch_size", type=int, default=1, help="Train batch size per device.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps.")
    parser.add_argument("--epochs", type=int, default=1, help="Train epochs.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--lora_rank", type=int, default=64, help="The dimension of the LoRA update matrices.")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA Alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/Qwen2.5-VL-3B-COT-sft-DA30-2",
        help="The output directory where the model predictions and checkpoints will be written."
    )
    args = parser.parse_args()

    # Load tokenizer and processor
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.pretrained_model)

    # Load pretrained model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.pretrained_model,
        torch_dtype=torch.bfloat16,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.pretrained_model,
        torch_dtype=torch.bfloat16,
    )

    train_ds = get_coco_sft_dataset()
    train_dataset = train_ds.map(process_func)

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,  # for train
        r=args.lora_rank,  # Lora rank
        lora_alpha=args.lora_alpha,  # Lora alaph
        lora_dropout=args.lora_dropout,  # Dropout
        bias="none",
    )

    # peft_model = get_peft_model(model, config)
    peft_model = model

    args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=10,
        logging_first_step=5,
        num_train_epochs=args.epochs,
        save_steps=1000,
        learning_rate=args.learning_rate,
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to="none",
        bf16=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    # peft_model, trainer = accelerator.prepare(peft_model, trainer)

    trainer.train()
    trainer.save_model(args.output_dir)