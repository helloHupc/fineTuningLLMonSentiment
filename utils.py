import numpy as np
import pandas as pd
import os
import json
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score,
                             classification_report,
                             confusion_matrix)
from transformers import EvalPrediction
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from peft import LoraConfig, PeftConfig, get_peft_model
from transformers import TrainerCallback, TrainingArguments, Trainer


def evaluate(y_true, y_pred, save_trained_folder, flag='before'):
    labels = ['positive', 'negative']
    mapping = {'positive': 1, 'negative': 0}
    def map_func(x):
        return mapping.get(x, 1)

    y_true = np.vectorize(map_func)(y_true)
    y_pred = np.vectorize(map_func)(y_pred)

    # Calculate accuracy
    all_accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    print(f'Accuracy: {all_accuracy:.5f}')

    # Generate classification report
    class_report = classification_report(y_true=y_true, y_pred=y_pred, digits=4, output_dict=True)

    # 提取所需的f1-score
    f1_score_0 = class_report['0']['f1-score']
    f1_score_1 = class_report['1']['f1-score']
    f1_score_macro = class_report['macro avg']['f1-score']
    f1_score_weighted = class_report['weighted avg']['f1-score']

    print('\nClassification Report:')
    print(classification_report(y_true=y_true, y_pred=y_pred, digits=4))

    # 准备保存结果
    results = {
        flag+"_eval_accuracy": all_accuracy,
        flag+"_f1_score_0": f1_score_0,
        flag+"_f1_score_1": f1_score_1,
        flag+"_f1_score_macro": f1_score_macro,
        flag+"_f1_score_weighted": f1_score_weighted
    }

    # 记录日志
    with open(f"{save_trained_folder}/all_results.json", "a") as file:
        json.dump(results, file, indent=4)

    return {"accuracy": all_accuracy, "f1": f1_score_macro}


def generate_prompt(data_point):
    return f"""generate_prompt
            Analyze the sentiment of the text enclosed in square brackets, 
            determine if it is positive, or negative, and return the answer as 
            the corresponding sentiment label "positive" or "negative"

            [{data_point["text"]}] = {data_point["label"]}
            """.strip()


def generate_test_prompt(data_point):
    return f"""
            Analyze the sentiment of the text enclosed in square brackets, 
            determine if it is positive, or negative, and return the answer as 
            the corresponding sentiment label "positive" or "negative"

            [{data_point["text"]}] = 

            """.strip()


# 模型存储路径
def get_save_trained_folder(model_name, filename):
    real_model_name = model_name.split("/")[1]
    dataset_cate_name = filename.split("/")[1]
    save_trained_folder = f"save_trained_model/{dataset_cate_name}-{real_model_name}"

    # 检查路径是否存在，如果不存在则创建
    if not os.path.exists(save_trained_folder):
        os.makedirs(save_trained_folder, exist_ok=True)

    return save_trained_folder


# 处理数据集
def process_dataset(filename, train_size, test_size, eval_size):
    df = pd.read_csv(filename,
                     names=["label", "text"],
                     encoding="utf-8", encoding_errors="replace")

    x_train = list()
    x_test = list()
    for label in ["positive", "negative"]:
        train, test = train_test_split(df[df.label == label],
                                       train_size=train_size,
                                       test_size=test_size,
                                       random_state=42)
        x_train.append(train)
        x_test.append(test)

    x_train = pd.concat(x_train).sample(frac=1, random_state=10)
    x_test = pd.concat(x_test)

    eval_idx = [idx for idx in df.index if idx not in list(train.index) + list(test.index)]
    x_eval = df[df.index.isin(eval_idx)]

    x_eval = (x_eval
              .groupby('label', group_keys=False)
              .apply(lambda x: x.sample(n=eval_size, random_state=10, replace=True)))
    x_train = x_train.reset_index(drop=True)

    x_train = pd.DataFrame(x_train.apply(generate_prompt, axis=1),
                           columns=["text"])
    x_eval = pd.DataFrame(x_eval.apply(generate_prompt, axis=1),
                          columns=["text"])

    y_true = x_test.label
    x_test = pd.DataFrame(x_test.apply(generate_test_prompt, axis=1), columns=["text"])

    train_data = Dataset.from_pandas(x_train)
    eval_data = Dataset.from_pandas(x_eval)

    return {
        'train': train_data,
        'eval': eval_data,
        'test': x_test,
        'y_true': y_true,
    }

def train_process_dataset(filename):
    df = pd.read_csv(filename, names=["label", "text"], encoding="utf-8", encoding_errors="replace")

    # 不进行分割，直接返回整个数据集
    x_train = df.sample(frac=1, random_state=10)  # 打乱数据顺序

    # 应用生成提示的函数（如果有的话）
    x_train = pd.DataFrame(x_train.apply(generate_prompt, axis=1), columns=["text"])

    train_data = Dataset.from_pandas(x_train)

    return {
        'train': train_data,
    }

# 处理T5需要的数据集
def deal_dataset(filename, train_size=0.8, eval_size=0.1, test_size=0.1):
    data = pd.read_csv(filename, header=None, names=["label", "text"])

    # 分割数据集
    # 首先将数据分成训练集和临时数据集（包含验证集和测试集）
    train_data, temp_data = train_test_split(data, test_size=(1 - train_size), random_state=42)

    # 然后将临时数据集再分成验证集和测试集
    eval_data, test_data = train_test_split(temp_data, test_size=test_size / (test_size + eval_size), random_state=42)

    train_data['text'] = train_data.apply(generate_prompt, axis=1)
    eval_data['text'] = eval_data.apply(generate_test_prompt, axis=1)
    test_data['text'] = test_data.apply(generate_test_prompt, axis=1)

    # 将数据转换为HuggingFace的Dataset格式
    train_dataset = Dataset.from_pandas(train_data)
    test_dataset = Dataset.from_pandas(test_data)
    eval_dataset = Dataset.from_pandas(eval_data)

    # 创建DatasetDict
    return DatasetDict({
        "train": train_dataset,
        "test": test_dataset,
        "eval": eval_dataset
    })


# T5数据集预处理函数
def preprocess_function(sample, tokenizer, max_seq_length):
    # Tokenize the texts
    max_seq_length = min(max_seq_length, tokenizer.model_max_length)
    model_inputs = tokenizer(
        sample["text"],
        padding="max_length",
        max_length=max_seq_length,
        truncation=True
    )

    labels = tokenizer(
        sample["label"],
        padding=True
    )

    model_inputs["labels"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    ]
    return model_inputs


def seq2seq_compute_metrics(tokenizer, metric, f1_metric):
    def compute_metrics(eval_pred: EvalPrediction):
        nonlocal tokenizer, metric, f1_metric
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        predictions, labels = postprocess_text(predictions, labels)
        result = metric.compute(predictions=predictions, references=labels)

        # 计算f1
        f1_result = f1_metric.compute(predictions=predictions, references=labels, average="weighted")

        # 合并结果
        result = {
            "accuracy": result["accuracy"],
            "f1": f1_result["f1"],
        }

        return result

    return compute_metrics


def postprocess_text(predictions, labels):
    label2id = {
        'negative': '0',
        'positive': '1'
    }
    predictions = [prediction.strip() for prediction in predictions]
    labels = [label2id[label.strip()] for label in labels]

    for idx in range(len(predictions)):
        if predictions[idx] in label2id:
           predictions[idx] = label2id[predictions[idx]]
        else:
            predictions[idx] = '-100'
    return predictions, labels


def get_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    return info.used//1024**2


def prepare_lora_model(model):
    peft_config = LoraConfig(
        lora_alpha=32,
        lora_dropout=0.05,
        r=8,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
        target_modules=["q", "v", "k"],
    )
    lora_model = get_peft_model(model, peft_config)
    return lora_model

