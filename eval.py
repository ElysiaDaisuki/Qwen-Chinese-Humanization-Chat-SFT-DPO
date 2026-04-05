import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline
)
from peft import PeftModel
from evaluate import load
from bert_score import score as bert_score
import numpy as np

# 基础模型路径
BASE_MODEL_PATH = "./Qwen2.5-7B-Instruct"
# LoRA 微调权重路径
LORA_MODEL_PATH = "./output/best_model"
#  BERT 模型路径（网络无法连接时使用本地bert）
LOCAL_BERT_PATH = "./bert-base-chinese"  # bert-base-uncased亦可
# 测试集路径
TEST_DATA_PATH = "./data/test_data.json"  
# 结果保存路径
SAVE_RESULT_PATH = "./eval_result/eval_result.json"
# 生成参数
GEN_MAX_LENGTH = 1024
GEN_TEMPERATURE = 0.1
TOP_P = 0.95

# 加载测试集
def load_test_data(data_path):
    """加载测试集，测试集要求格式：
    [
    {"input":"用户问题","target":"标准答案"},
    ...
    ]
    """
    if data_path.endswith(".jsonl"):
        with open(data_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
    else:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    inputs = [item["input"] for item in data]
    targets = [item["target"] for item in data]
    return inputs, targets

# 加载 LoRA 微调后的模型 
def load_lora_model(base_model_path, lora_path):
    # 加载base模型
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16  # 已修复弃用警告
    )

    # 加载 LoRA 权重
    model = PeftModel.from_pretrained(model, lora_path)
    model = model.merge_and_unload()  # 合并权重

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token

    # 生成 pipeline
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=GEN_MAX_LENGTH,
        temperature=GEN_TEMPERATURE,
        top_p=TOP_P,
        do_sample=True,
        repetition_penalty=1.1
    )
    return generator, tokenizer

# 模型批量生成回答 
def generate_answers(generator, inputs):
    """模型生成预测回答"""
    predictions = []
    print("开始模型生成回答...")
    for query in tqdm(inputs, desc="Generating"):
        # 对话模板
        prompt = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
        response = generator(prompt)[0]["generated_text"]
        # 截取模型回答部分
        pred = response.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()
        predictions.append(pred)
    return predictions

# 评估指标计算 
def calculate_distinct(predictions):
    #计算模糊度
    unigrams = []
    bigrams = []
    for sent in predictions:
        tokens = sent.split()
        unigrams.extend(tokens)
        bigrams.extend(zip(tokens, tokens[1:]))
    
    distinct_1 = len(set(unigrams)) / len(unigrams) if len(unigrams) > 0 else 0
    distinct_2 = len(set(bigrams)) / len(bigrams) if len(bigrams) > 0 else 0
    return {"distinct-1": round(distinct_1, 4), "distinct-2": round(distinct_2, 4)}

def evaluate_metrics(predictions, targets, local_bert_path):
    #计算评估指标
    print("开始计算评估指标（BERTScore + 模糊度）...")
    metrics_result = {}

    # BERTScore
    P, R, F1 = bert_score(
        predictions, targets,
        model_type=local_bert_path,
        num_layers=12,
        verbose=False,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    metrics_result["bertscore_precision"] = round(P.mean().item(), 4)
    metrics_result["bertscore_recall"] = round(R.mean().item(), 4)
    metrics_result["bertscore_f1"] = round(F1.mean().item(), 4)

    # Distinct 模糊度
    distinct = calculate_distinct(predictions)
    metrics_result.update(distinct)

    return metrics_result

# 主函数 
if __name__ == "__main__":
    os.makedirs("./eval_result", exist_ok=True)  # <-- 必须加
    
    # 加载数据
    inputs, targets = load_test_data(TEST_DATA_PATH)
    
    # 加载模型
    generator, tokenizer = load_lora_model(BASE_MODEL_PATH, LORA_MODEL_PATH)
    
    # 生成预测
    predictions = generate_answers(generator, inputs)
    
    # 计算评估指标
    eval_results = evaluate_metrics(predictions, targets, LOCAL_BERT_PATH)
    
    # 保存详细结果
    final_output = {
        "eval_metrics": eval_results,
        "samples": [
            {"input": i, "target": t, "prediction": p}
            for i, t, p in zip(inputs, targets, predictions)
        ]
    }
    
    with open(SAVE_RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
    
    # 打印核心指标
    print("\n" + "="*50)
    print("模型评估完成")
    print(f"BERTScore-F1: {eval_results['bertscore_f1']}")
    print(f"模糊度 Distinct-1: {eval_results['distinct-1']}  Distinct-2: {eval_results['distinct-2']}")
    print("="*50)