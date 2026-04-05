# Qwen2.5-7B-Instruct 网络闲聊风格微调与评估
该项目基于LCCC网络闲聊数据实现了对阿里云通义千问 Qwen2.5-7B-Instruct 模型的 LoRA 微调、推理和评估功能，使模型学会网络聊天对话风格，更具人性。支持其他形式的对话数据的训练、批量推理和多维度指标评估。

## 概览

- [基础模型](#项目结构) - [项目结构](#项目结构) - [项目环境](#项目环境) - [核心功能说明](#核心功能说明) - [快速开始](#快速开始) - [注意事项](#注意事项) - [常见问题](#常见问题) - [许可证](#许可证)

## 基础模型

本项目基于阿里云通义千问 [Qwen2.5-7B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct)实现

  

## 项目结构

solution/

  

├── config.py # 模型训练/推理配置

  

├── dataset.py # 数据集加载与处理

  

├── eval.py # 模型评估（BERTScore + Distinct）

  

├── inference.py # 模型推理（交互式/单条/批量）

  

├── train.py # 模型训练主程序

 
├── utils.py # 工具函数（模型保存、种子设置等）

├── data/ 

│   ├── train.json
│   ├── valid.json
│   └── test.json
├── eval_result/ # 评估结果目录（自动生成）
└── Qwen2.5-7B-Instruct/ # 基础模型目录


## 项目环境

终端使用指令下载相关依赖:

```pip install -r requirements.txt```

### 硬件要求

-   训练：建议至少 1 张 A10/A100/V100 GPU（≥16GB 显存）
-   推理：支持 CPU/GPU，GPU 推理需≥8GB 显存
  
## 核心功能说明
### 1. 数据集处理

-   自动清理文本（移除中文字符间多余空格）
-  无效对话清洗(过短对话或空对话)
-   支持按比例采样数据集（快速验证）
-   适配 Qwen2.5 chat template 格式

### 2. LoRA 微调

-   目标模块：q_proj/k_proj/v_proj/o_proj
-   支持梯度累积、学习率预热
-   自动保存最佳验证损失模型

### 3. 推理优化

-   支持批量推理（提升效率）
-   可配置温度、最大生成长度等参数
-   支持对话历史管理（交互式模式）

### 4. 评估体系

-   BERTScore：衡量生成文本与标准答案的语义相似度
-   Distinct：衡量生成文本的多样性
  

## 快速开始

  

### 1. 数据准备
训练 / 验证集格式：JSON 数组，每个元素为对话列表
```
[
	[
	"话说红海真的好看！！！准备二刷了！！！但求多排场！！！",
	"我明天二刷 ！",
	"必须的！上班之后可能就没空了呢 ~"
	],
	[
	"上厕所是别玩手机！除非他是防水的",
	"说出你的故事"
	]
]
```
测试集格式（评估用）：
```
[ 
	{"input": "用户问题", "target": "标准答案"}, 
	{"input": "今天吃什么", "target": "可以试试吃点家常菜，比如番茄炒蛋"} 
]
```
### 2. 模型训练
基础训练命令
```
python train.py \ 
	--model_name_or_path ./Qwen2.5-7B-Instruct \ 
	--train_file ./data/... \ 
	--valid_file ./data/... \ 
	--output_dir ./output \ 
	--use_lora \ 
	--num_train_epochs 3 \ 
	--per_device_train_batch_size 8 \ 
	--learning_rate 2e-4
 ``` 
关键参数说明
| 参数 | 类型 | 说明 | 默认值 | 
| :--- | :--- | :--- | :--- | 
| --model_name_or_path | str | 预训练模型路径 | ./Qwen-2.5-7B-Instruct |
| --train_file | str | 训练集文件路径 | ./data/path to train set.json | 
| --valid_file | str | 验证集文件路径 | ./data/path to valid set.json |
 | --output_dir | str | 模型输出目录 | ./output |
  | --train_ratio | float | 训练集使用比例 (0-1) | 1.0 |
   | --valid_ratio | float | 验证集使用比例 (0-1) | 1.0 |
  | --max_length | int | 最大序列长度 | 1024 | 
  | --num_train_epochs | int | 训练轮数 | 3 | 
| --per_device_train_batch_size | int | 单卡训练批大小 | 8 | 
| --per_device_eval_batch_size | int | 单卡验证批大小 | 4 | 
| --gradient_accumulation_steps | int | 梯度累积步数 | 1 |
| --learning_rate | float | 学习率 | 2e-4 |
| --warmup_steps | int | 学习率预热步数 | 100 |
| --use_lora | action | 是否使用 LoRA 微调 | True | 
| --lora_r | int | LoRA 秩 | 16 |
| --lora_alpha | int | LoRA alpha 参数 | 16 |
| --fp16 | action | 是否使用 FP16 混合精度 | False |
| --bf16 | action | 是否使用 BF16 混合精度 | True |
| --seed | int | 随机种子 | 11 |
 #### 训练输出

-   最佳模型：`./output/best_model`
-   最终模型：`./output/final_model`
-   训练配置：`./output/args.json`
### 3. 模型推理
交互式对话
```
python inference.py \
  --base_model ./Qwen2.5-7B-Instruct \
  --lora_path ./output/best_model \
  --mode interactive
```
单条推理
```
python inference.py \ 
	--base_model ./Qwen2.5-7B-Instruct \ 
	--lora_path ./output/best_model \ 
	--mode single \ 
	--prompt "你好，介绍一下自己"
```
批量推理
```
python inference.py \
	--base_model ./Qwen2.5-7B-Instruct \
	--lora_path ./output/best_model \
	--mode batch \
	--input_file ./data/test_data.json \
	--output_file ./outputs/responses.json
```
### 4. 模型评估
使用如下指令进行评估
```python eval.py```
#### 评估指标

-   **BERTScore**：Precision/Recall/F1（基于中文 BERT 模型）
-   **Distinct**：
    
    -   Distinct-1：单字 / 词唯一性
    -   Distinct-2：双字 / 词组合唯一性
    

#### 评估输出

-   详细结果：`./eval_result/eval_result.json`
-   控制台输出核心指标：
```
==================================================
模型评估完成
BERTScore-F1: xxxxx
模糊度 Distinct-1: xxxxx  Distinct-2: xxxxx
==================================================
```
## 注意事项
-   模型下载：
    -   Qwen2.5-7B-Instruct 模型需从阿里云模型库下载
    -   本地 BERT 模型（bert-base-chinese）用于网络不畅时离线计算 BERTScore
    
-   数据格式：
    
    -   确保 JSON 文件编码为 UTF-8
    -   双人对话效果最佳（用户 - 助手交替）
    
-   训练优化：
    
    -   若显存不足，可减小`per_device_train_batch_size`或增大`gradient_accumulation_steps`
    -   建议使用 bf16（需 GPU 支持）加速训练
## 常见问题

-   **训练时显存不足**：
    ```
    # 减小批大小
    python train.py --per_device_train_batch_size 4 --gradient_accumulation_steps 2
    
    ```
-   **评估时 BERTScore 报错**：
    
    -   确保本地 BERT 模型路径正确（`LOCAL_BERT_PATH`）
    -   网络可用时可直接使用`model_type="bert-base-chinese"`
    
-   **推理结果重复**：
    
    -   降低 temperature 值（如 0.1）
    -   增大 repetition_penalty（如 1.2）
    - 
     ## 许可证
     本项目仅用于研究目的，使用需遵循 Qwen 模型的开源许可证。