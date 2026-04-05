import os
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel, LoraConfig, get_peft_model
from config import default_config

# ===================== 全局配置 =====================
class Config:
    """DPO超参数"""
    # 模型与数据路径
    MODEL_PATH = default_config.base_model_path
    SFT_MODEL_PATH = default_config.merge_model_path
    DATA_PATH = default_config.dpo_data
    OUTPUT_DIR = default_config.dpo_output_dir

    # 训练超参数
    MAX_LENGTH = default_config.dpo_max_length
    BATCH_SIZE = default_config.dpo_batch_size
    GRADIENT_ACCUMULATION_STEPS = default_config.dpo_gradient_accumulation_steps
    LEARNING_RATE = default_config.dpo_learning_rate
    EPOCHS = default_config.dpo_epochs
    BETA = default_config.dpo_beta
    USE_DATA_SIZE = default_config.dpo_use_data_size

    # LoRA配置
    LORA_R = default_config.dpo_lora_r
    LORA_ALPHA = default_config.dpo_lora_alpha
    LORA_DROPOUT = default_config.dpo_lora_dropout
    LORA_TARGET_MODULES = ["q_proj", "v_proj"]

    # 训练精度
    MIXED_PRECISION = default_config.dpo_mixed_precision
    DEVICE = default_config.device

# ===================== 工具函数 =====================
def create_quantization_config() -> BitsAndBytesConfig:
    """创建4bit量化配置"""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

def get_log_probabilities(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor
) -> torch.Tensor:
    """计算序列的对数概率"""
    
    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)
    labels = labels.to(model.device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=False,
        return_dict=True
    )
    logits = outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid_mask = (shift_labels != -100)

    log_probs = nn.functional.log_softmax(shift_logits, dim=-1)
    shift_labels_valid = shift_labels.masked_fill(~valid_mask, 0)
    log_probs = log_probs.gather(-1, shift_labels_valid.unsqueeze(-1)).squeeze(-1)
    log_probs = log_probs * valid_mask.float()

    return log_probs.sum(dim=-1)

def compute_dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1
) -> torch.Tensor:
   
    log_ratio = (policy_chosen_logps - ref_chosen_logps) - (policy_rejected_logps - ref_rejected_logps)
    return -nn.functional.logsigmoid(beta * log_ratio).mean()

# ===================== 数据集 =====================
class DPODataset(Dataset):
    """DPO训练数据集"""
    def __init__(self, data_path: str, tokenizer, max_length: int, use_size: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id

        # 加载数据
        with open(data_path, encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        self.data = self._process_data(lines[:use_size])

    def _process_data(self, lines: list) -> list:
        """解析并格式化对话数据"""
        processed = []
        for line in tqdm(lines, desc="Processing dataset"):
            data = json.loads(line)
            prompt = self._build_prompt(data["context"])
            chosen = data["chosen"]["text"].strip()
            rejected = data["rejected"]["text"].strip()

            if not all([prompt, chosen, rejected]):
                continue

            # 编码并拼接数据
            item = self._encode_sample(prompt, chosen, rejected)
            processed.append(item)

        return processed

    def _build_prompt(self, context: list) -> str:
        """构建对话prompt"""
        prompt = ""
        for turn in context:
            text = turn["text"].strip()
            if not text:
                continue
            if turn["role"] == "human":
                prompt += f"<|im_start|>user\n{text}<|im_end|>\n"
            else:
                prompt += f"<|im_start|>assistant\n{text}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt

    def _pad_truncate_left(self, ids: list, max_len: int, pad_val: int) -> list:
        """填充/截断"""
        if len(ids) > max_len:
            ids = ids[-max_len:]
        pad_length = max_len - len(ids)
        return [pad_val] * pad_length + ids

    def _encode_sample(self, prompt: str, chosen: str, rejected: str) -> dict:
        """编码单条偏好样本"""
        # 编码
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        chosen_ids = self.tokenizer(chosen, add_special_tokens=False)["input_ids"] + [self.tokenizer.eos_token_id]
        rejected_ids = self.tokenizer(rejected, add_special_tokens=False)["input_ids"] + [self.tokenizer.eos_token_id]

        # 拼接完整序列
        chosen_full = prompt_ids + chosen_ids
        rejected_full = prompt_ids + rejected_ids

        # 填充与截断
        c_input_ids = self._pad_truncate_left(chosen_full, self.max_length, self.pad_token_id)
        r_input_ids = self._pad_truncate_left(rejected_full, self.max_length, self.pad_token_id)

        # 构建标签
        c_labels = [-100] * len(prompt_ids) + chosen_ids
        r_labels = [-100] * len(prompt_ids) + rejected_ids
        c_labels = self._pad_truncate_left(c_labels, self.max_length, -100)
        r_labels = self._pad_truncate_left(r_labels, self.max_length, -100)

        # 构建attention mask
        c_mask = [1 if i != self.pad_token_id else 0 for i in c_input_ids]
        r_mask = [1 if i != self.pad_token_id else 0 for i in r_input_ids]

        return {
            "c_input_ids": torch.LongTensor(c_input_ids),
            "c_mask": torch.LongTensor(c_mask),
            "c_labels": torch.LongTensor(c_labels),
            "r_input_ids": torch.LongTensor(r_input_ids),
            "r_mask": torch.LongTensor(r_mask),
            "r_labels": torch.LongTensor(r_labels),
        }

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        return self.data[idx]

# ===================== 模型构建 =====================
def build_tokenizer(model_path: str) -> AutoTokenizer:
    """构建tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    return tokenizer

def build_policy_model(config: Config) -> PeftModel:
    """构建训练模型（policy model）"""
    bnb_config = create_quantization_config()
    model = AutoModelForCausalLM.from_pretrained(
        config.SFT_MODEL_PATH,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False
    )

    # LoRA配置
    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        target_modules=config.LORA_TARGET_MODULES,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 训练配置
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.train()
    return model

def build_reference_model(config: Config) -> AutoModelForCausalLM:
    """构建参考模型（reference model）"""
    bnb_config = create_quantization_config()
    model = AutoModelForCausalLM.from_pretrained(
        config.SFT_MODEL_PATH,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model

# ===================== 训练逻辑 =====================
def train_one_epoch(
    policy_model: PeftModel,
    ref_model: AutoModelForCausalLM,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    config: Config,
    pbar: tqdm
):
    """单轮训练逻辑"""
    step = 0
    for batch in dataloader:
        # 数据加载到GPU
        c_ids = batch["c_input_ids"].to(config.DEVICE)
        c_mask = batch["c_mask"].to(config.DEVICE)
        c_labels = batch["c_labels"].to(config.DEVICE)
        r_ids = batch["r_input_ids"].to(config.DEVICE)
        r_mask = batch["r_mask"].to(config.DEVICE)
        r_labels = batch["r_labels"].to(config.DEVICE)

        # 计算参考模型概率
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=config.MIXED_PRECISION):
            ref_chosen = get_log_probabilities(ref_model, c_ids, c_mask, c_labels)
            ref_rejected = get_log_probabilities(ref_model, r_ids, r_mask, r_labels)

        # 计算策略模型概率与损失
        with torch.cuda.amp.autocast(enabled=config.MIXED_PRECISION):
            policy_chosen = get_log_probabilities(policy_model, c_ids, c_mask, c_labels)
            policy_rejected = get_log_probabilities(policy_model, r_ids, r_mask, r_labels)
            loss = compute_dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, config.BETA)
            loss = loss / config.GRADIENT_ACCUMULATION_STEPS

        # 反向传播
        scaler.scale(loss).backward()

        # 梯度累积更新
        if (step + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # 更新进度条
        pbar.update(1)
        current_loss = loss.item() * config.GRADIENT_ACCUMULATION_STEPS
        pbar.set_postfix(loss=f"{current_loss:.4f}")
        step += 1

# ===================== 主函数 =====================
def main():
    """DPO训练主流程"""
    # 初始化配置
    config = Config()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    torch.manual_seed(42)  # 固定随机种子

    # 构建tokenizer
    print("Loading tokenizer...")
    tokenizer = build_tokenizer(config.MODEL_PATH)

    # 构建数据集与加载器
    print("Preparing dataset...")
    dataset = DPODataset(
        data_path=config.DATA_PATH,
        tokenizer=tokenizer,
        max_length=config.MAX_LENGTH,
        use_size=config.USE_DATA_SIZE
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=True
    )

    # 构建模型
    print("Building models...")
    policy_model = build_policy_model(config)
    ref_model = build_reference_model(config)

    # 初始化优化器与混合精度
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=config.LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler(enabled=config.MIXED_PRECISION)

    # 开始训练
    print("Starting DPO training...")
    total_steps = config.EPOCHS * len(dataloader)
    with tqdm(total=total_steps, desc="Training") as pbar:
        for epoch in range(config.EPOCHS):
            print(f"\n========== Epoch {epoch + 1}/{config.EPOCHS} ==========")
            train_one_epoch(
                policy_model=policy_model,
                ref_model=ref_model,
                dataloader=dataloader,
                optimizer=optimizer,
                scaler=scaler,
                config=config,
                pbar=pbar
            )

    # 保存模型
    print(f"Saving model to {config.OUTPUT_DIR}...")
    policy_model.save_pretrained(config.OUTPUT_DIR)
    tokenizer.save_pretrained(config.OUTPUT_DIR)

    print("DPO training complete")

if __name__ == "__main__":
    main()