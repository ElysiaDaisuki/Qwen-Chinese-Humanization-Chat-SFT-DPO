import json
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any
import random
import logging

logger = logging.getLogger(__name__)


class ConversationDataset(Dataset):
    """对话数据集"""
    
    def __init__(self, file_path: str, tokenizer, max_length: int = 512, ratio: float = 1.0):
        """
        初始化数据集
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载数据
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} conversations from {file_path}")
        
        # 根据比例采样数据
        if ratio < 1.0:
            sample_size = int(len(data) * ratio)
            data = random.sample(data, sample_size)
            logger.info(f"Sampled {sample_size} conversations (ratio={ratio})")
        
        self.data = self._process_data(data)
        logger.info(f"Processed {len(self.data)} samples")
    
    def _process_data(self, data: List[List[str]]) -> List[Dict[str, Any]]:
        """处理原始数据，构建对话格式"""
        processed_data = []
        
        for conversation in data:
            # 跳过空对话
            if not conversation or len(conversation) < 2:
                continue
            
            # 构建对话消息列表
            messages = []
            for i, text in enumerate(conversation):
                # 清理文本中的多余空格
                text = self._clean_text(text)
                
                #一般是两人对话，让引起谈话的人当用户，回话的人当助手
                if i % 2 == 0:
                    messages.append({"role": "user", "content": text})
                else:
                    messages.append({"role": "assistant", "content": text})
            
            # 使用 chat template 构建完整文本
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False  
            )
            
            processed_data.append({"text": text, "messages": messages})
        
        return processed_data
    
    def _clean_text(self, text: str) -> str:
        """清理文本中的多余空格"""
        if not text:
            return text
        
        # 移除中文字符间的多余空格
        import re
        text = re.sub(r'([\u4e00-\u9fff])\s+([\u4e00-\u9fff])', r'\1\2', text)
        text = re.sub(r'([\u4e00-\u9fff])\s+([，。！？；：""''、])', r'\1\2', text)
        text = re.sub(r'([，。！？；：""''、])\s+([\u4e00-\u9fff])', r'\1\2', text)
        
        return text.strip()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        
        # 标签与输入相同（用于语言建模）
        labels = input_ids.clone()
        # 对于padding部分，设置标签为-100（忽略损失）
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def collate_fn(batch):
    """拼接成完整的batch"""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


# 添加一个用于验证的数据集类
class ValidationDataset(Dataset):
    """验证数据集 - 用于计算评估指标"""
    
    def __init__(self, file_path: str, tokenizer, max_length: int = 512, ratio: float = 1.0):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if ratio < 1.0:
            sample_size = int(len(data) * ratio)
            data = random.sample(data, sample_size)
        
        self.data = []
        for conversation in data:
            if len(conversation) >= 2:
                # 取最后一轮对话
                user_msg = conversation[-2]
                assistant_msg = conversation[-1]
                
                # 清理文本
                user_msg = self._clean_text(user_msg)
                assistant_msg = self._clean_text(assistant_msg)
                
                self.data.append({
                    "user": user_msg,
                    "assistant": assistant_msg
                })
        
        logger.info(f"Loaded {len(self.data)} validation samples")
    
    def _clean_text(self, text: str) -> str:
        #同训练数据
        if not text:
            return text
        
        import re
        text = re.sub(r'([\u4e00-\u9fff])\s+([\u4e00-\u9fff])', r'\1\2', text)
        text = re.sub(r'([\u4e00-\u9fff])\s+([，。！？；：""''、])', r'\1\2', text)
        text = re.sub(r'([，。！？；：""''、])\s+([\u4e00-\u9fff])', r'\1\2', text)
        
        return text.strip()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return item["user"], item["assistant"]