import numpy as np
import pandas as pd
import os
from tqdm import tqdm

import torch
import torch.nn as nn

import transformers
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          BitsAndBytesConfig,
                          TrainingArguments,
                          pipeline,
                          logging)
from datasets import Dataset
from peft import LoraConfig, PeftConfig
import bitsandbytes as bnb
from trl import SFTTrainer
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split
import utils
import warnings

warnings.filterwarnings("ignore")

model_name = "google/gemma-1.1-2b-it"

compute_dtype = getattr(torch, "float16")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
)

model.config.use_cache = False
model.config.pretraining_tp = 1

max_seq_length = 256
tokenizer = AutoTokenizer.from_pretrained(model_name, max_seq_length=max_seq_length)
EOS_TOKEN = tokenizer.eos_token

# dataset
filename = "data/FinancialNews/data.csv"
# filename = "data/ChnSentiCorp/data.csv"
# filename = "data/WeiboSentiment/data.csv"
# filename = "data/TwitterSentiment/data.csv"

processed_data = utils.process_dataset(filename, 200, 200, 200)

train_data = processed_data['train']
eval_data = processed_data['eval']
test_data = processed_data['test']
y_true = processed_data['y_true']

print("train_data len:", len(train_data))
print("Training data sample:", train_data[0])
print("Eval data sample:", eval_data[0])
print("eval_data len:", len(eval_data))
print("test_data len:", len(test_data))

def predict(test_data, model, tokenizer):
    y_pred = []
    for i in tqdm(range(len(test_data))):
        prompt = test_data.iloc[i]["text"]
        input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(**input_ids, max_new_tokens=1, temperature=0.0)
        result = tokenizer.decode(outputs[0])
        answer = result.split("=")[-1].lower()
        if "positive" in answer:
            y_pred.append("positive")
        elif "negative" in answer:
            y_pred.append("negative")
        else:
            y_pred.append("none")
    return y_pred


save_trained_folder = utils.get_save_trained_folder(model_name, filename)

# 模型未进行微调的表现
y_pred = predict(test_data, model, tokenizer)

utils.evaluate(y_true, y_pred, save_trained_folder)

# 微调训练器 SFTTrainer
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj", ],
)

training_arguments = TrainingArguments(
    output_dir=save_trained_folder+"/logs",
    num_train_epochs=5,
    gradient_checkpointing=True,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    save_steps=0,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=False,
    eval_strategy='epoch',
    eval_steps=112,
    eval_accumulation_steps=2,
    lr_scheduler_type="linear",
    report_to="tensorboard",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    peft_config=peft_config,
    dataset_text_field="text",
    tokenizer=tokenizer,
    max_seq_length=max_seq_length,
    args=training_arguments,
    packing=False,
)

# Train model
trainer.train()

# Save trained model
# trainer.model.save_pretrained(save_trained_folder)

y_pred = predict(test_data, model, tokenizer)
metrics = utils.evaluate(y_true, y_pred, save_trained_folder, 'after')
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)

summary_write_log_dir = trainer.args.output_dir
writer = SummaryWriter(log_dir=summary_write_log_dir)
writer.add_scalar("eval/accuracy", metrics['accuracy'])
writer.add_scalar("eval/f1", metrics['f1'])
writer.close()
