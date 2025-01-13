from datasets import load_dataset,load_from_disk
import re
from transformers import AutoTokenizer,AutoModelForCausalLM,DataCollatorForLanguageModeling
from transformers import TrainingArguments
from trl import SFTTrainer,DataCollatorForCompletionOnlyLM
from trl.trainer import ConstantLengthDataset
from peft import get_peft_model, PromptTuningConfig, TaskType, PromptTuningInit, PeftModel, PeftConfig
import torch
import pdb
import argparse
import torch
import numpy as np
import random
import os


def train_model(
    data_path, 
    model_path, 
    output_path, 
    seed=2024,
    num_epochs=3,
    batch_size=4,
    learning_rate=2e-5,
    max_length=1024,
    weight_decay=0.01,
    warmup_ratio=0.1,
    gradient_accumulation_steps=1
):
    """
    Train a model using the specified paths and hyperparameters.
    
    Args:
        data_path (str): Path to the training data
        model_path (str): Path to the base model
        output_path (str): Path for saving the output
        seed (int): Random seed for reproducibility
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size per device during training
        learning_rate (float): Learning rate for training
        max_length (int): Maximum sequence length
        weight_decay (float): Weight decay for optimization
        warmup_ratio (float): Ratio of warmup steps
        gradient_accumulation_steps (int): Number of steps for gradient accumulation
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    def set_seed(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    set_seed(seed)
    
    # ... formatting functions remain the same ...
    def formatting_prompts_func(examples):
        output_text = []
        for i in range(len(examples["instruction"])):
            instruction = examples["instruction"][i]
            response = examples["output"][i]
            text = f'''### Instruction:{instruction}\n### Response:{response}'''
            output_text.append(text)
        return output_text
    
    # Load dataset and model
    dataset = load_from_disk(data_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        use_cache=False)
#        attn_implementation="flash_attention_2"
#    )
    
    tokenizer.pad_token = tokenizer.eos_token
    
    # Training arguments
    args = TrainingArguments(
        output_dir=output_path,
        weight_decay=weight_decay,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        logging_steps=10,
        save_strategy="no",
        learning_rate=learning_rate,
        bf16=True,
        tf32=True,
        max_grad_norm=1.0,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        max_seq_length=max_length,
        tokenizer=tokenizer,
        formatting_func=formatting_prompts_func,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        dataset_kwargs={
            "add_special_tokens": False,
            "append_concat_token": False,
        }
    )
    
    # Train and save
    trainer.train()
    trainer.model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    del model,tokenizer,trainer
#    import torch
    torch.cuda.empty_cache()

# Run garbage collector
    import gc
    gc.collect()
    
    return 

def train_status_check(trained_model_path):
    """
    Check if the trained model folder has identical .safetensors files
    and tokenizer.json as the base model folder.
    
    Args:
        base_model_path (str): Path to the base model directory
        trained_model_path (str): Path to the trained model directory
    
    Returns:
        bool: True if both folders have identical files, False otherwise
    """
    # Check if trained model path exists
    if not os.path.exists(trained_model_path):
        return False
    
    # Get all .safetensors files in both directories
    #base_safetensors = {f for f in os.listdir(base_model_path) if f.endswith('.safetensors')}
    trained_safetensors = {f for f in os.listdir(trained_model_path) if f.endswith('.safetensors')}
    
    # Check for tokenizer.json in both directories
    #base_has_tokenizer = os.path.exists(os.path.join(base_model_path, 'tokenizer.json'))
    trained_has_tokenizer = os.path.exists(os.path.join(trained_model_path, 'tokenizer.json'))
    
    # Check if the sets of safetensors files are identical
    #safetensors_match = base_safetensors == trained_safetensors
    
    # Return True only if all conditions are met
    return check_safetensors_files(trained_safetensors) and trained_has_tokenizer

def check_safetensors_files(files):
    """
    Check if the given set of files contains all safetensor files from 1 to m,
    where m is dynamically determined from the filenames.

    Args:
    - files (set): A set of filenames.

    Returns:
    - bool: True if all files from 1 to m are present, False otherwise.
    """
    # Extract the maximum value of m from the filenames
    pattern = re.compile(r"model-\d{5}-of-(\d{5})\.safetensors")
    m_values = {int(match.group(1)) for file in files if (match := pattern.match(file))}
    
    # Ensure there's only one unique m value in the filenames
    if len(m_values) != 1:
        return False
    
    m = m_values.pop()  # Get the unique value of m
#    print(m)
    # Generate the expected filenames
    expected_files = {f"model-{i:05d}-of-{m:05d}.safetensors" for i in range(1, m + 1)}
    
    # Check if the set contains all the expected files
    return expected_files.issubset(files)
