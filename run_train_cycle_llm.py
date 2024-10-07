from tqdm import tqdm
import torch
import os
import numpy as np
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          BitsAndBytesConfig,
                          TrainingArguments)
from peft import LoraConfig
from trl import SFTTrainer
import utils
import warnings
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")

model_name = "./save_trained_model/WeiboSentiment-Qwen2-1.5B-Instruct"

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
    trust_remote_code=True
)

model.config.use_cache = False
model.config.pretraining_tp = 1

max_seq_length = 256
tokenizer = AutoTokenizer.from_pretrained(model_name, max_seq_length=max_seq_length, trust_remote_code=True)
EOS_TOKEN = tokenizer.eos_token

# dataset
filename = "data/DynamicWeibo/data-2.csv"

processed_data = utils.train_process_dataset(filename)

train_data = processed_data['train']

print("train_data len:", len(train_data))
print("Training data sample:", train_data[0])


save_trained_folder = 'save_cycle_model/WeiboSentiment-Qwen2-1.5B-Instruct-cycle-2'
if not os.path.exists(save_trained_folder):
    os.makedirs(save_trained_folder, exist_ok=True)


# 微调训练器 SFTTrainer
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
)


training_arguments = TrainingArguments(
    output_dir=save_trained_folder+"/logs",
    num_train_epochs=5,
    gradient_checkpointing=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
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
    eval_strategy='no',
)


trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
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
trainer.model.save_pretrained(save_trained_folder)
tokenizer.save_pretrained(save_trained_folder)
trainer.model.config.save_pretrained(save_trained_folder)