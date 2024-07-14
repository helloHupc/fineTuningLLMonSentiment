## 模型微调代码

> 《基于社交媒体情感分析的品牌声誉舆情管理研究》

------

此项目是论文中使用的模型微调代码，共有4个[数据集](#数据集)，3个LLM大模型，分别是[Qwen2](https://github.com/QwenLM/Qwen2?tab=readme-ov-file)、[Gemma](https://github.com/google-deepmind/gemma)、[Llama2](https://github.com/meta-llama/llama)



#### 运行环境

代码运行在**RTX4060Ti 16G**加**32G**内存的本地环境。模型微调参数需要根据实际硬件环境进行调整。

项目依赖扩展参见：[requirements.txt](requirements.txt)



#### 数据集

代码中使用的4个数据集都是用的开源数据集，并且都经过了预处理和统一格式化处理。处理后的数据在data目录下。

下面表格展示了处理后的数据对应的开源数据集。

| 处理前数据集名称                                             | 处理后项目中数据集名称 |
| ------------------------------------------------------------ | ---------------------- |
| [dirtycomputer/weibo_senti_100k](https://huggingface.co/datasets/dirtycomputer/weibo_senti_100k) | WeiboSentiment         |
| [MrbBakh/Twitter_Sentiment](https://huggingface.co/datasets/MrbBakh/Twitter_Sentiment)       | TwitterSentiment       |
| [ChnSentiCorp句子级情感分类数据集](https://www.luge.ai/#/luge/dataDetail?id=25) | ChnSentiCorp           |
| [zeroshot/twitter-financial-news-sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) | FinancialNews          |

#### 启动微调

```bash
# gemma-1.1-2b-it 微调脚本
python run_gemma.py

# Llama-2-7b-hf 微调脚本
python run_llama2.py

# Qwen2-1.5B-Instruct 微调脚本
python run_qwen.py
```



#### 微调后的模型调用示例
```bash
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

```

```
outputs
Predictions: negative
```


