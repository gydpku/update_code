import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
model_name="/dccstor/obsidian_llm/yiduo/models--Qwen--QwQ-32B-Preview"
#model = AutoModelForCausalLM.from_pretrained(
#        model_name,
#        torch_dtype="auto",
#        device_map="auto"
#    )
#tokenizer = AutoTokenizer.from_pretrained(model_name)
def generate_response(model,tokenizer,prompt,model_name="/dccstor/obsidian_llm/yiduo/models--Qwen--QwQ-32B-Preview",temperature=0.2, max_new_tokens=2048):
    # Load model and tokenizer

    # Prepare chat messages
    messages = [
        {"role": "system", "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."},
        {"role": "user", "content": prompt}
    ]

    # Format the messages using the tokenizer's chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize the input text
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Generate the response
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature
    )

    # Extract the generated response (excluding the input tokens)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Decode the response into text
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

# Example usage
#model_name = "/dccstor/obsidian_llm/yiduo/models--Qwen--QwQ-32B-Preview"
#prompt = "How many r in strawberry."
#response = generate_response(model_name, prompt)
#print(response)


def read_jsonl(file_path):
    """
    Reads a JSONL file and returns a list of dictionaries.
    
    Args:
        file_path (str): Path to the .jsonl file.
        
    Returns:
        list: A list of dictionaries, each representing a JSON object from the file.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))  # Parse each line as a JSON object
    return data
def load_a1_a2_a3_data(path_a1='/dccstor/obsidian_llm/yiduo/summary/src/a1.jsonl',path_a2='/dccstor/obsidian_llm/yiduo/summary/src/a2.jsonl',path_a3='/dccstor/obsidian_llm/yiduo/summary/src/a3.jsonl'):
    a1_data=[]
    with open(path_a1,'r') as f:
        for line in f:
            a1_data.append(json.loads(line))
    a2_data=[]
    with open(path_a2,'r') as f:
        for line in f:
            a2_data.append(json.loads(line))
    a3_data=[]
    with open(path_a3,'r') as f:
        for line in f:
            a3_data.append(json.loads(line))
    return a1_data,a2_data,a3_data
def simple_process_a1_a2_a3_data(a1_data,a2_data,a3_data):
    a1_prompts_answers=[]   
    for data in a1_data:
        a1_prompts_answers.append({'Input':data['query'],'Output':data['std'],'COT_Output':data['cot']})
    a2_prompts_answers=[]
    for data in a2_data:
        a2_prompts_answers.append({'Input':data['query'],'Output':data['std'],'COT_Output':data['cot']})
    a3_prompts_answers=[]
    for data in a3_data:
        a3_prompts_answers.append({'Input':data['query'],'Output':data['std'],'COT_Output':data['cot']})
    return a1_prompts_answers,a2_prompts_answers,a3_prompts_answers

def collect_simple_solutions(a1_prompts_answers,a2_prompts_answers,a3_prompts_answers,solution_nums=32,temperature=0.2):
    a1_prompts_answers_with_solutions=[]
    print('Loading model...')
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print('Processing prompts...')
    for data in a1_prompts_answers:
        prompt=data['Input']
        solutions=[]
        for i in range(solution_nums):
            print('a1',i)
            solution=generate_response(model,tokenizer,prompt,temperature=temperature)
            solutions.append(solution)
        data['Solutions']=solutions
        data['Temperature']=temperature
        a1_prompts_answers_with_solutions.append(data)
    a2_prompts_answers_with_solutions=[]
    for data in a2_prompts_answers:
        prompt=data['Input']
        solutions=[]
        for i in range(solution_nums):
            print('a2',i)
            solution=generate_response(model,tokenizer,prompt,temperature=temperature)
            solutions.append(solution)
        data['Solutions']=solutions
        data['Temperature']=temperature
        a2_prompts_answers_with_solutions.append(data)
    a3_prompts_answers_with_solutions=[]
    for data in a3_prompts_answers:
        prompt=data['Input']
        solutions=[]
        for i in range(solution_nums):
            print('a3',i)
            solution=generate_response(model,tokenizer,prompt,temperature=temperature)
            solutions.append(solution)
        data['Solutions']=solutions
        data['Temperature']=temperature
        a3_prompts_answers_with_solutions.append(data)
    return a1_prompts_answers_with_solutions,a2_prompts_answers_with_solutions,a3_prompts_answers_with_solutions

import os
path='/dccstor/obsidian_llm/yiduo'
temperature=0.2
print('Loading data...')
a1_data,a2_data,a3_data=load_a1_a2_a3_data()
print('Simple prompting...')
a1_prompts_answers,a2_prompts_answers,a3_prompts_answers=simple_process_a1_a2_a3_data(a1_data,a2_data,a3_data)
print('Solution generating...')
a1_prompts_answers_with_solutions,a2_prompts_answers_with_solutions,a3_prompts_answers_with_solutions=collect_simple_solutions(a1_prompts_answers,a2_prompts_answers,a3_prompts_answers,solution_nums=32,temperature=temperature)
print('Writing file')
with open(os.path.join(path,'a1_prompts_answers_with_solutions_{0}.jsonl'.format(temperature)),'w') as f:
    for data in a1_prompts_answers_with_solutions:
        f.write(json.dumps(data)+'\n')
with open(os.path.join(path,'a2_prompts_answers_with_solutions_{0}.jsonl'.format(temperature)),'w') as f:
    for data in a2_prompts_answers_with_solutions:
        f.write(json.dumps(data)+'\n')
with open(os.path.join(path,'a3_prompts_answers_with_solutions_{0}.jsonl'.format(temperature)),'w') as f:
    for data in a3_prompts_answers_with_solutions:
        f.write(json.dumps(data)+'\n')
