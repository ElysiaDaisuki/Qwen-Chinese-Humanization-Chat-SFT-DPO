import os
import json
import logging
import argparse
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    set_seed
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from config import default_config
from dataset import ConversationDataset, collate_fn
from utils import save_model

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# 启用SDPA相关优化
torch.backends.cuda.enable_flash_sdp(True)  
torch.backends.cuda.enable_math_sdp(True)   
torch.backends.cuda.enable_mem_efficient_sdp(True)  

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Qwen2.5-7B-Instruct微调训练")
    
    # 模型和数据路径
    parser.add_argument("--model_name_or_path", type=str, 
                        default=default_config.model_name_or_path,
                        help="模型路径")
    parser.add_argument("--train_file", type=str, 
                        default=default_config.train_file,
                        help="训练集文件")
    parser.add_argument("--valid_file", type=str, 
                        default=default_config.valid_file,
                        help="验证集文件")
    parser.add_argument("--output_dir", type=str, 
                        default=default_config.output_dir,
                        help="输出目录")
    
    # 数据比例控制 
    parser.add_argument("--train_ratio", type=float, 
                        default=default_config.train_ratio,
                        help="训练集使用比例 (0-1)")
    parser.add_argument("--valid_ratio", type=float, 
                        default=default_config.valid_ratio,
                        help="验证集使用比例 (0-1)")
    
    # 训练超参数
    parser.add_argument("--max_length", type=int, 
                        default=default_config.max_length,
                        help="最大序列长度")
    parser.add_argument("--num_train_epochs", type=int, 
                        default=default_config.num_train_epochs,
                        help="训练轮数")
    parser.add_argument("--per_device_train_batch_size", type=int, 
                        default=default_config.per_device_train_batch_size,
                        help="训练批大小")
    parser.add_argument("--per_device_eval_batch_size", type=int, 
                        default=default_config.per_device_eval_batch_size,
                        help="验证批大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, 
                        default=default_config.gradient_accumulation_steps,
                        help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, 
                        default=default_config.learning_rate,
                        help="学习率")
    parser.add_argument("--warmup_steps", type=int, 
                        default=default_config.warmup_steps,
                        help="预热步数")
    
    # LoRA参数
    parser.add_argument("--use_lora", action="store_true", 
                        default=default_config.use_lora,
                        help="是否使用LoRA")
    parser.add_argument("--lora_r", type=int, 
                        default=default_config.lora_r,
                        help="LoRA秩")
    parser.add_argument("--lora_alpha", type=int, 
                        default=default_config.lora_alpha,
                        help="LoRA alpha")
    
    # 其他
    parser.add_argument("--fp16", action="store_true", 
                        default=default_config.fp16,
                        help="使用混合精度")
    parser.add_argument("--bf16", action="store_true", 
                        default=default_config.bf16,
                        help="使用bf16混合精度")
    parser.add_argument("--seed", type=int, 
                        default=default_config.seed,
                        help="随机种子")
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    # 打印配置信息
    logger.info("=" * 50)
    logger.info("训练配置:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 50)
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存配置
    with open(os.path.join(args.output_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    
    # 加载模型和tokenizer
    logger.info(f"加载模型: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.bf16:
        torch_dtype = torch.bfloat16
        logger.info("使用 bf16 混合精度训练")
    elif args.fp16:
        torch_dtype = torch.float16
        logger.info("使用 fp16 混合精度训练")
    else:
        torch_dtype = torch.float32
        logger.info("使用 fp32 训练")

    model_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
        "device_map": "auto",
        "attn_implementation": "sdpa",  # 启用SDPA
        "use_cache": False,  # 训练时禁用cache
    }
    if args.fp16:
        model_kwargs["low_cpu_mem_usage"] = True
        scaler = torch.cuda.amp.GradScaler()
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **model_kwargs
    )

    
    # LoRA
    if args.use_lora:
        logger.info("配置LoRA...")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=default_config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    model.gradient_checkpointing_enable()
    
    # 按比例加载数据集
    logger.info(f"加载训练集 (使用比例: {args.train_ratio})...")
    train_dataset = ConversationDataset(
        args.train_file,
        tokenizer,
        max_length=args.max_length,
        ratio=args.train_ratio  # 控制训练集使用比例
    )
    logger.info(f"训练集实际大小: {len(train_dataset)}")
    
    logger.info(f"加载验证集 (使用比例: {args.valid_ratio})...")
    valid_dataset = ConversationDataset(
        args.valid_file,
        tokenizer,
        max_length=args.max_length,
        ratio=args.valid_ratio  # 控制验证集使用比例
    )
    logger.info(f"验证集实际大小: {len(valid_dataset)}")
    
    # 创建DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
        collate_fn=collate_fn
    )
    
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # 优化器和调度器
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    # 训练循环
    logger.info("开始训练...")
    global_step = 0
    best_loss = float("inf")
    
    for epoch in range(args.num_train_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
        
        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["labels"].cuda()
            
            # 前向传播
            if args.bf16:
                # bf16 使用 autocast
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss / args.gradient_accumulation_steps
                loss.backward()
                
            elif args.fp16:
                with torch.cuda.amp.autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss / args.gradient_accumulation_steps
                scaler.scale(loss).backward()
                
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss / args.gradient_accumulation_steps
                loss.backward()
            
            total_loss += loss.item() * args.gradient_accumulation_steps
            
            # 梯度累积
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                progress_bar.set_postfix({
                    "loss": f"{total_loss / global_step:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}"
                })
        
        # 每个epoch结束验证
        model.eval()
        eval_loss = 0
        with torch.no_grad():
            for batch in valid_dataloader:
                input_ids = batch["input_ids"].cuda()
                attention_mask = batch["attention_mask"].cuda()
                labels = batch["labels"].cuda()
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                eval_loss += outputs.loss.item()
        
        eval_loss /= len(valid_dataloader)
        logger.info(f"Epoch {epoch + 1} - eval_loss: {eval_loss:.4f}")
        
        # 保存最佳模型
        if eval_loss < best_loss:
            best_loss = eval_loss
            save_model(model, tokenizer, args.output_dir, "best_model")
            logger.info(f"保存最佳模型，loss: {best_loss:.4f}")
    
    # 保存最终模型
    save_model(model, tokenizer, args.output_dir, "final_model")
    logger.info(f"训练完成，模型保存至: {args.output_dir}")


if __name__ == "__main__":
    main()