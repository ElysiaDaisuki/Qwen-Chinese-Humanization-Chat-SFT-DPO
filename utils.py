import os
import json
import torch
from typing import Dict, Any, Tuple
import numpy as np


def save_model(model, tokenizer, output_dir: str, name: str = "model"):
    """保存模型"""
    #创建目录
    os.makedirs(output_dir, exist_ok=True) 
    # 模型
    model.save_pretrained(os.path.join(output_dir, name))
    # tokenizer
    tokenizer.save_pretrained(os.path.join(output_dir, name))
    
    print(f"模型已保存至: {os.path.join(output_dir, name)}")

def set_seed(seed: int):
    """不使用transformer手动设置随机种子(可选)"""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)