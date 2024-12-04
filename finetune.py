from datasets import load_dataset,load_from_disk
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


# Argument parser to accept parameters from command-line
parser = argparse.ArgumentParser(description="Script for transferring paths")

parser.add_argument('--data_path', type=str, required=True, help="Path to the data",default='/dccstor/obsidian_llm/yiduo/summary/src/2k_op_valid_3_re_sql')
parser.add_argument('--model_path', type=str, required=True, help="Path to the model",default='/dccstor/obsidian_llm/yiduo/deepseek-coder-6.7b-base')
parser.add_argument('--output_path', type=str, required=True, help="Path for output",default='/dccstor/obsidian_llm/yiduo/copy_v2/finetuned_models/2k_op_valid_3_re_sql_model')

# Parse arguments
args = parser.parse_args()

data_path = args.data_path
model_path = args.model_path
output_path = args.output_path
tokenizer = AutoTokenizer.from_pretrained(model_path)
seed=2024

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(seed)
'''
peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init=PromptTuningInit.TEXT,
    prompt_tuning_init_text="Generate inspirational quotes",  # Provides a starter for the model to begin searching for the best embeddings
#    num_virtual_tokens=3,  # This doesn't have to match the length of the text above
    tokenizer_name_or_path=model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    use_cache = False)
'''
#text_peft_model = get_peft_model(model, peft_config)
#pdb.set_trace()
def prompt_no_input(row):
    return ("Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:\n{output}").format_map(row)


def prompt_input(row):
    return ("Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}").format_map(row)
def create_alpaca_prompt(row):
    return prompt_no_input(row)
#"### Response:"
dataset = load_from_disk(data_path)
#pdb.set_trace()
#dataset = dataset.shuffle(seed=seed)
def formatting_prompts_func(examples):
    output_text = []
    for i in range(len(examples["instruction"])):
        instruction = examples["instruction"][i]
        response = examples["output"][i]
        text = f'''### Instruction:{instruction}\n### Response:{response}'''
        output_text.append(text)
    return output_text
instruction_template ='### Instruction:\n'
#tokenizer.add_special_tokens({'unk_token': '<RES>'})
#response_template_with_context = '<|begin_of_text|>'
#response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]
#pdb.set_trace()
#dataset = dataset.map(create_conversation, remove_columns=dataset.features,batched=False)
#dataset = dataset.train_test_split(test_size=2)
#dataset["train"].to_json("train_dataset.json", orient="records")
#dataset = load_dataset("json", data_files="train_dataset.json", split="train")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    use_cache = False,
    attn_implementation="flash_attention_2")
#tokenizer = AutoTokenizer.from_pretrained(model_path)
#pdb.set_trace()
#tokenizer.add_special_tokens({'pad_token': '[PAD]'})
'''
peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init=PromptTuningInit.RANDOM,
    num_virtual_tokens=20,
    tokenizer_name_or_path=model_path
)
'''
# Set the pad_token_id
#tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')
#tokenizer.pad_token = '[PAD]'
tokenizer.pad_token = tokenizer.eos_token
#text_peft_model = get_peft_model(model, peft_config)
#response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False) #[2:]
args = TrainingArguments(
    output_dir=output_path, # directory to save and repository id
    weight_decay=0.01,
    num_train_epochs=3,                     # number of training epochs
    per_device_train_batch_size=4,          # batch size per device during training
    gradient_accumulation_steps=1,          # number of steps before performing a backward/update pass
    gradient_checkpointing=True,            # use gradient checkpointing to save memory
    optim="adamw_torch_fused",              # use fused adamw optimizer
    logging_steps=10,                       # log every 10 steps
    save_strategy="no",                  # save checkpoint every epoch
    learning_rate=2e-5,                     # learning rate, based on QLoRA paper
    bf16=True,                              # use bfloat16 precision
    tf32=True,                              # use tf32 precision
    max_grad_norm=1.0,                      # max gradient norm based on QLoRA paper
    warmup_ratio=0.1,                      # warmup ratio based on QLoRA paper
    lr_scheduler_type="cosine",           # use constant learning rate scheduler
)
max_seq_length = 1024
#collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer) #collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template, tokenizer=tokenizer, mlm=False) #collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
#    packing=True,
    formatting_func=formatting_prompts_func, #create_alpaca_prompt,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    dataset_kwargs={
        "add_special_tokens": False,  # We template with special tokens
        "append_concat_token": False, # No need to add additional separator token
    }
)
trainer.train() 
# save model
trainer.model.save_pretrained(output_path) #save_model() #output_path+'/2_epoch')
tokenizer.save_pretrained(output_path)
'''
data_path='/dccstor/obsidian_llm/yiduo/three-shot-hug' 
dataset = load_from_disk(data_path)
dataset = dataset.shuffle()
pdb.set_trace()
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    formatting_func=create_alpaca_prompt,
    dataset_kwargs={
        "add_special_tokens": False,  # We template with special tokens
        "append_concat_token": False, # No need to add additional separator token
    }
)
trainer.train(resume_from_checkpoint=output_path+'/checkpoint-786')
trainer.save_model(output_path+'/4_epoch')
data_path='/dccstor/obsidian_llm/yiduo/zero-shot-hug' 
dataset = load_from_disk(data_path)
dataset = dataset.shuffle()
args.num_train_epochs = 1
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    formatting_func=create_alpaca_prompt,
    dataset_kwargs={
        "add_special_tokens": False,  # We template with special tokens
        "append_concat_token": False, # No need to add additional separator token
    }
)
trainer.train(resume_from_checkpoint=output_path+'/checkpoint-786')
trainer.save_model(output_path)
'''
del model
del trainer
torch.cuda.empty_cache()
