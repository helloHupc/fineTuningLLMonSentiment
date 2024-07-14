import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import bitsandbytes as bnb
import torch
import torch.nn as nn
import transformers
from datasets import Dataset
from peft import LoraConfig, PeftConfig
from trl import SFTTrainer
from trl import setup_chat_format
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          BitsAndBytesConfig,
                          TrainingArguments,
                          pipeline,
                          logging)
import utils
import warnings

warnings.filterwarnings("ignore")

model_name = "meta-llama/Llama-2-7b-hf"

compute_dtype = getattr(torch, "float16")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map='auto',
    torch_dtype=compute_dtype,
    quantization_config=bnb_config,
)

model.config.use_cache = False
model.config.pretraining_tp = 1

max_seq_length = 512
tokenizer = AutoTokenizer.from_pretrained(model_name, max_seq_length=max_seq_length)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# dataset
filename = "data/FinancialNews/data.csv"
# filename = "data/ChnSentiCorp/data.csv"
# filename = "data/WeiboSentiment/data.csv"
# filename = "data/TwitterSentiment/data.csv"

processed_data = utils.process_dataset(filename, 200, 100, 100)

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
        pipe = pipeline(task="text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        max_new_tokens=1,
                        temperature=0.0,
                        do_sample=False
                        )
        result = pipe(prompt)
        answer = result[0]['generated_text'].split("=")[-1]
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

# # 微调训练器 SFTTrainer
#
# peft_config = LoraConfig(
#     lora_alpha=16,
#     lora_dropout=0.1,
#     r=64,
#     bias="none",
#     task_type="CAUSAL_LM",
#     target_modules="all-linear",
# )
#
#
# training_arguments = TrainingArguments(
#     output_dir=save_trained_folder+"/logs",  # directory to save and repository id
#     num_train_epochs=3,  # number of training epochs
#     per_device_train_batch_size=2,  # batch size per device during training
#     gradient_accumulation_steps=8,  # number of steps before performing a backward/update pass
#     gradient_checkpointing=True,  # use gradient checkpointing to save memory
#     optim="paged_adamw_32bit",
#     save_steps=0,
#     logging_steps=25,  # log every 10 steps
#     learning_rate=2e-4,  # learning rate, based on QLoRA paper
#     weight_decay=0.001,
#     fp16=True,
#     bf16=False,
#     max_grad_norm=0.3,  # max gradient norm based on QLoRA paper
#     max_steps=-1,
#     warmup_ratio=0.03,  # warmup ratio based on QLoRA paper
#     group_by_length=False,
#     lr_scheduler_type="cosine",  # use cosine learning rate scheduler
#     report_to="tensorboard",  # report metrics to tensorboard
#     evaluation_strategy="epoch"  # save checkpoint every epoch
# )
#
# trainer = SFTTrainer(
#     model=model,
#     args=training_arguments,
#     train_dataset=train_data,
#     eval_dataset=eval_data,
#     peft_config=peft_config,
#     dataset_text_field="text",
#     tokenizer=tokenizer,
#     max_seq_length=max_seq_length,
#     packing=False,
#     dataset_kwargs={
#         "add_special_tokens": False,
#         "append_concat_token": False,
#     }
# )
#
# # Train model
# trainer.train()
#
# # Save trained model
# trainer.model.save_pretrained(save_trained_folder)
#
# y_pred = predict(test_data, model, tokenizer)
# utils.evaluate(y_true, y_pred, save_trained_folder)
