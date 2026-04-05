import os
import json
import torch
import argparse
from typing import List, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from config import default_config 


class QwenInference:
    def __init__(
        self,
        base_model_path: str=default_config.base_model_path, 
        lora_path: str = default_config.lora_path,
        device: str = default_config.device,
        max_length: int = default_config.inference_max_length,
        system_prompt: str = "你是一个智能助手，请根据用户的问题提供准确、有帮助的回答。"
    ):
        
        self.device = device
        self.max_length = max_length
        self.system_prompt = system_prompt
        
        #tokenizer构建
        print(f"Loading tokenizer from {base_model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            padding_side="left",       
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        #基础模型读取
        print(f"Loading model from {base_model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        #加载lora
        if lora_path and os.path.exists(lora_path):
            print(f"Loading LoRA adapter from {lora_path}...")
            self.model = PeftModel.from_pretrained(self.model, lora_path)
        
        self.model.eval()
        print(f"Model loaded on {self.device}")
    
    #内容生成
    def generate(
        self,
        user_input: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        repetition_penalty: float = 1.1,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:

        # 构建消息列表
        messages = []
        
        # 添加系统提示
        sys_prompt = system_prompt or self.system_prompt
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        
        # 添加历史对话
        if history:
            messages.extend(history)
        
        # 添加当前用户输入
        messages.append({"role": "user", "content": user_input})
        
        # 应用聊天模板
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Tokenize输入
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                top_k=top_k if do_sample else 50,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=repetition_penalty,
            )
        
        # 解码回复
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        
        return response.strip()
    
    def chat(self):
        """交互式对话"""
        print("\n" + "=" * 60)
        print("Interactive Chat Mode")
        print("Commands:")
        print("  /quit - Exit")
        print("  /clear - Clear conversation history")
        print("  /temp [value] - Set temperature")
        print("  /max [value] - Set max_new_tokens")
        print("=" * 60)
        
        history = []
        temperature = 0.7
        max_new_tokens = 512
        
        while True:
            user_input = input("\n[You]: ").strip()
            
            if not user_input:
                continue
            
            # 处理命令
            if user_input.lower() == '/quit':
                print("Goodbye!")
                break
            elif user_input.lower() == '/clear':
                history = []
                print("Conversation history cleared.")
                continue
            elif user_input.lower().startswith('/temp'):
                try:
                    temperature = float(user_input.split()[1])
                    print(f"Temperature set to {temperature}")
                except:
                    print(f"Current temperature: {temperature}")
                continue
            elif user_input.lower().startswith('/max'):
                try:
                    max_new_tokens = int(user_input.split()[1])
                    print(f"Max new tokens set to {max_new_tokens}")
                except:
                    print(f"Current max_new_tokens: {max_new_tokens}")
                continue
            
            # 生成回复
            response = self.generate(
                user_input,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                history=history,
            )
            
            print(f"[AI]: {response}")
            
            # 保存历史
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})
            
            # 限制历史长度
            if len(history) > 20:
                history = history[-20:]
    
    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        batch_size: int = 8,
    ) -> List[str]:
        
        """批量回复"""
        responses = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            
            # 构建消息列表
            texts = []
            for prompt in batch_prompts:
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ]
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                texts.append(text)
            
            # Tokenize
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True,
            ).to(self.device)
            
            # 生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # 解码
            for j, output in enumerate(outputs):
                input_len = inputs.input_ids[j].shape[0]
                response = self.tokenizer.decode(
                    output[input_len:],
                    skip_special_tokens=True,
                )
                responses.append(response.strip())
            
            print(f"Processed batch {i//batch_size + 1}/{(len(prompts)-1)//batch_size + 1}")
        
        return responses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True, help="Base model path")
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA adapter path")
    parser.add_argument("--mode", type=str, default="interactive", 
                        choices=["interactive", "single", "batch"], help="Inference mode")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt for single mode")
    parser.add_argument("--input_file", type=str, default=None, help="Input file for batch mode")
    parser.add_argument("--output_file", type=str, default="./outputs/responses.json", 
                        help="Output file for batch mode")
    parser.add_argument("--max_new_tokens", type=int, default=default_config.inference_max_length, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=default_config.inference_temperature, help="Temperature")
    
    args = parser.parse_args()
    
    # 初始化推理器
    inferencer = QwenInference(
        base_model_path=args.base_model,
        lora_path=args.lora_path,
    )
    
    if args.mode == "interactive":
        # 交互式对话
        inferencer.chat()
    
    elif args.mode == "single":
        # 单轮推理
        if not args.prompt:
            print("Error: Please provide --prompt for single mode")
            return
        
        response = inferencer.generate(
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        print(f"\n[User]: {args.prompt}")
        print(f"[AI]: {response}")
    
    elif args.mode == "batch":
        # 批量推理
        if not args.input_file:
            print("Error: Please provide --input_file for batch mode")
            return
        
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取prompts
        prompts = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    # 从conversations中提取用户输入
                    if "conversations" in item:
                        for conv in item["conversations"]:
                            if conv.get("from") == "human":
                                prompts.append(conv.get("value", ""))
                                break
                    elif "input" in item:
                        prompts.append(item["input"])
                    elif "prompt" in item:
                        prompts.append(item["prompt"])
                elif isinstance(item, str):
                    prompts.append(item)
        
        print(f"Total prompts: {len(prompts)}")
        
        # 批量生成
        responses = inferencer.batch_generate(
            prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        
        # 保存结果
        results = []
        for prompt, response in zip(prompts, responses):
            results.append({
                "input": prompt,
                "output": response,
            })
        
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()