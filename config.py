# config.py
from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class ModelConfig:
    """模型配置 - 作为默认值"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 模型路径
    model_name_or_path: str = "./Qwen2.5-7B-Instruct"
    
    # 数据集路径
    train_file: str = "./data/LCCC-base_train_clean.json"
    valid_file: str = "./data/LCCC-base_valid_clean.json"
    test_file: str = "./data/LCCC-base_test_clean.json"
    
    # 训练参数（默认值）
    max_length: int = 512
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    
    # 数据比例控制
    train_ratio: float = 1.0
    valid_ratio: float = 1.0
    
    # LoRA参数
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    
    # 其他
    output_dir: str = "./output"
    logging_dir: str = "./logs"
    save_steps: int = 4000
    eval_steps: int = 4000
    logging_steps: int = 50
    
    #精度选择
    bf16: bool = True
    fp16: bool = False

    #随机种子
    seed: int = 42

    merge_model_path="./sft_model"
        
    #推理参数
    base_model_path="./Qwen2.5-7B-Instruct"
    lora_path="./output/best_model"
    inference_max_length:int = 512
    inference_temperature:float = 0.7

    # DPO参数
    dpo_data="./data/rlhf/harmless_base_cn_train.jsonl"
    
    dpo_batch_size: int = 4  
    dpo_gradient_accumulation_steps: int = 2  
    dpo_learning_rate: float = 2e-6  
    dpo_epochs: int = 1  
    dpo_beta: float = 0.1  
    dpo_use_data_size: int = 4000  
    dpo_mixed_precision: bool = True 
    dpo_max_length: int=512

    dpo_lora_r:int=8
    dpo_lora_alpha:int=16
    dpo_lora_dropout:float=0.05

    dpo_output_dir="./dpo_final_model"

# 创建默认配置实例
default_config = ModelConfig()