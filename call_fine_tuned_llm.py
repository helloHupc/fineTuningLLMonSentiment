from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
from tqdm import tqdm
import utils
import warnings

warnings.filterwarnings("ignore")

# 定义保存模型的文件夹路径
save_trained_folder = "./save_trained_model/WeiboSentiment-Qwen2-1.5B-Instruct"

# 加载本地微调好的模型和分词器
model = AutoModelForCausalLM.from_pretrained(save_trained_folder)
tokenizer = AutoTokenizer.from_pretrained(save_trained_folder)

# 移动模型到GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义预测函数
def predict(test_data, model, tokenizer):
    y_pred = []
    for i in tqdm(range(len(test_data))):
        prompt = test_data.iloc[i]["text"]
        input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(**input_ids, max_new_tokens=1, temperature=0.001, do_sample=True)
        result = tokenizer.decode(outputs[0])
        answer = result.split("=")[-1].lower()
        if "positive" in answer:
            return "positive"
        elif "negative" in answer:
            return "negative"
        else:
            return "none"


# 示例句子
sentence = "如果是在国外：处罚小米恶意营销带来网络效应，罚款1000万？国内处理就随随便便找个个人顶包完事了"

# 将句子转化为 DataFrame
sentence_df = pd.DataFrame({"text": [sentence]})

# 应用 generate_test_prompt 函数格式化句子
formatted_sentence = sentence_df.apply(utils.generate_test_prompt, axis=1)

# 将格式化后的句子转化为 DataFrame，模拟 x_test 的格式
formatted_sentence_df = pd.DataFrame(formatted_sentence, columns=["text"])

print(formatted_sentence_df)

# 使用加载的模型进行预测
predictions = predict(formatted_sentence_df, model, tokenizer)
print("Predictions:", predictions)
