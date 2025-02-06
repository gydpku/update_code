import os
from datasets import load_from_disk
import datasets
import re
import pdb
from vllm import LLM,SamplingParams

def get_names(path):
    return [folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path,folder))]

def load_model(model_name,temperature=0.0,max_new_tokens=1500):
    sampling_params = SamplingParams(temperature=temperature,max_tokens=max_new_tokens, top_p=0.95)
# load model
    llm = LLM(model_name,max_model_len=8192,gpu_memory_utilization=0.95)
    return llm,sampling_params

def generate(prompt,llm,sampling_params):
    response = llm.generate(
        prompt,
        sampling_params,
    )
    return response.choices[0].message.content
path='/dccstor/obsidian_llm/yiduo/summary/src/TinyZero/checkpoints/TinyZero/dentist_qa-Qwen2.5-7B-grpo/actor'

def calculate_dataset(dataset,llm,sampling_params):
    test_dataset=dataset['test']
    num=0
    count=0
    for data in test_dataset:
        query=data['query']
        response=generate(query,llm,sampling_params)
        try:
            gold=data['std']
        except:
            gold=data['answer']
        answer_pattern = r'<answer>(.*?)</answer>'
        match = re.finditer(answer_pattern, solution_str)
        matches = list(match)
        if matches:
            final_answer = matches[-1].group(1).strip()
        else:
            final_answer = None
        if final_answer is not None:
            if final_answer==gold:
                count+=1
        num+=1
        pdb.set_trace()
        return count,num,count/num
names=get_names(path)
name_performance={}
for name in names:
    model_path=os.path.join(path,name)
    llm,sparam=load_model(model_path)
    dataset=datasets.load_from_disk('/dccstor/obsidian_llm/yiduo/summary/src/medical_qa')
    count,num,acc=calculate_dataset(dataset,llm,sparam)
    name_performance[name]=acc
    print(name_performance)
