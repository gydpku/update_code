from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import torch
import os
import pdb
import time
import json
import torch
import os
import pdb
import time
import re
import copy
from collections import Counter
def extract_answer(text):
    matches = re.findall(r'\b[A-D]\b', text)
    if matches:
        return matches[0]
    return None
def extract_answer_solution_chinese(solution):
    solution_sentences=solution.split('\n\n')
    for sentence_id in range(len(solution_sentences)-1,-1,-1):
        sentence=solution_sentences[sentence_id]
        if extract_answer(sentence):
            return extract_answer(sentence)
    return None

def extract_answers_solutions(solutions):
    answers=[]
    for solution in solutions:
        answer=extract_answer_solution_chinese(solution)
        if answer:
            answers.append(answer)
    answer_counter=Counter(answers)
    most_common_answer=answer_counter.most_common(1)[0][0]
    most_common_answer_count=answer_counter.most_common(1)[0][1]
    return most_common_answer,answers
def calculate_accuracy(datas):
    correct_count=0
    for data in datas:
        label=data['Output']
        majority_answer,_=extract_answers_solutions(data['Solutions'])
        if majority_answer==label:
            correct_count+=1
    return correct_count/len(datas)
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
# Global initialization of model and tokenizer
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
def initialize_model_and_tokenizer(model_name):
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer
def random_sample(demos,num=4):
    import random
    sampled_demos=random.sample(demos,num)
    text='\nHere are some demonstration examples:'
    for example_id,example in enumerate(sampled_demos):
        text+='Example {0}: {1}'.format(example_id,example)
    return text
def do_simple_inference(prompt, model, tokenizer, temperature=0.2, max_new_tokens=2048):
    """
    Perform a single inference with the given prompt, model, and tokenizer.

    Args:
        prompt (str): The user prompt.
        model: The pre-trained model.
        tokenizer: The tokenizer corresponding to the model.
        temperature (float): Sampling temperature for text generation.
        max_new_tokens (int): Maximum number of new tokens to generate.

    Returns:
        str: The generated response.
    """
    # Define the chat context
    messages = [
        {"role": "system", "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."},
        {"role": "user", "content": prompt}
    ]

    # Convert messages to a single formatted text
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Tokenize input text
    model_inputs = tokenizer(text, return_tensors="pt", truncation=True).to(model.device)

    # Generate response
    print('Generating response...')
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature
    )

    # Decode and return the response
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return response
    

def generate_response(prompt, model, tokenizer,demos, solution_num=32, temperature=0.2, max_new_tokens=2048):
    start_time = time.time()
    
    #text=random_sample(demos,num=1) # Prepare messages for all solutions in one batch
    messages = [
        [
            {"role": "system", "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."},
            {"role": "user", "content": prompt+random_sample(demos,num=1)}
        ]
        for _ in range(solution_num)
    ]

    try:
        # Format all messages in one batch
        texts = [
            tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True
            ) for message in messages
        ]

        # Tokenize all texts in one batch
        model_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)

        print('Generating responses in batch')
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )

        # Decode responses in batch
        responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        torch.cuda.empty_cache()
       
    
    except Exception as e:
        print(f"Error occurred during generation: {e}")
        return []

    end_time = time.time()
    print('Time taken:', end_time - start_time)
#    pdb.set_trace()
    return responses
# Generate response function
# Data processing and solution generation
def collect_simple_solutions(a1_prompts_answers, a2_prompts_answers, a3_prompts_answers, model, tokenizer, solution_nums=32, temperature=0.2):
    
    '''
    data_solutions=[[] for iter in range(solution_nums//8)]
    for data in a1_prompts_answers:
        prompt="Here is a question and its solution. Your task is to rewrite the solution to make it detailed in your style and directly output the detailed solution."
        prompt+="Question:\n"+data['Input']
        prompt+="its solution:\n"+data['COT_Output']
        detailed_solution=do_simple_inference(prompt, model, tokenizer, temperature=0.2, max_new_tokens=2048).split('assistant')[-1]
        data_solutions.append((data['Input'],detailed_solution))
        print('Q:',data['Input'],'ori:',data['COT_Output'])
        print('new:',detailed_solution)
    '''
        
    a1_prompts_answers_with_solutions=[[] for iter in range(solution_nums//8)]
    for data in a1_prompts_answers:
        prompt = data['Input']
        data['Solutions'] =[]
        data['Temperature'] = temperature
        other_data_points = ['Input:'+d['Input']+'\nOutput:'+data['COT_Output'] for d in a1_prompts_answers if d != data]  #other_data_points = ['Input:'+d['Input']+'\nOutput:'+data['COT_Output'] for d in data_solutions if d[0] != data['Input']]
        print('a1', data['Input'], data['Output'], data['COT_Output'],len(a1_prompts_answers),len(other_data_points), '\n')
        for iter in range(solution_nums//8):
            solutions = generate_response(prompt, model, tokenizer,other_data_points, solution_num=8, temperature=temperature)
            data['Solutions']+=solutions
            a1_prompts_answers_with_solutions[iter].append(copy.deepcopy(data))

    for iter in range(solution_nums//8):
        print("a1_acc",iter,calculate_accuracy(a1_prompts_answers_with_solutions[iter]))
    a1_prompts_answers_with_solutions=a1_prompts_answers_with_solutions[-1]
    a2_prompts_answers_with_solutions = [[] for iter in range(solution_nums // 8)]
    for data in a2_prompts_answers:
        prompt = data['Input']
        data['Solutions'] = []
        data['Temperature'] = temperature
        other_data_points = ['Input:'+d['Input']+'\nOutput:'+data['COT_Output'] for d in a2_prompts_answers if d != data]  #other_data_points = ['Input:' + d['Input'] + '\nOutput:' + data['COT_Output'] for d in data_solutions if d[0] != data['Input']]
        print('a2', data['Input'], data['Output'], data['COT_Output'], len(a2_prompts_answers), len(other_data_points), '\n')
        for iter in range(solution_nums // 8):
            solutions = generate_response(prompt, model, tokenizer, other_data_points, solution_num=8, temperature=temperature)
            data['Solutions'] += solutions
            a2_prompts_answers_with_solutions[iter].append(copy.deepcopy(data))

    for iter in range(solution_nums // 8):
        print("a2_acc", iter, calculate_accuracy(a2_prompts_answers_with_solutions[iter]))
    a2_prompts_answers_with_solutions = a2_prompts_answers_with_solutions[-1]
    
    a3_prompts_answers_with_solutions = [[] for iter in range(solution_nums // 8)]
    for data in a3_prompts_answers:
        prompt = data['Input']
        data['Solutions'] = []
        data['Temperature'] = temperature
        other_data_points = ['Input:'+d['Input']+'\nOutput:'+data['COT_Output'] for d in a3_prompts_answers if d != data]  #other_data_points = ['Input:' + d['Input'] + '\nOutput:' + data['COT_Output'] for d in data_solutions if d[0] != data['Input']]
        print('a3', data['Input'], data['Output'], data['COT_Output'], len(a3_prompts_answers), len(other_data_points), '\n')
        for iter in range(solution_nums // 8):
            solutions = generate_response(prompt, model, tokenizer, other_data_points, solution_num=8, temperature=temperature)
            data['Solutions'] += solutions
            a3_prompts_answers_with_solutions[iter].append(copy.deepcopy(data))

    for iter in range(solution_nums // 8):
        print("a3_acc", iter, calculate_accuracy(a3_prompts_answers_with_solutions[iter]))
    a3_prompts_answers_with_solutions = a3_prompts_answers_with_solutions[-1]
   
    return a1_prompts_answers_with_solutions, a2_prompts_answers_with_solutions, a3_prompts_answers_with_solutions

# Main logic
if __name__ == "__main__":
    path = '/dccstor/obsidian_llm/yiduo'
    temperature = 0.7
    solution_nums = 64
    model_name = "/dccstor/obsidian_llm/yiduo/models--Qwen--QwQ-32B-Preview"
    type='in_context_major_64'
    print("Initializing model and tokenizer...")
    model, tokenizer = initialize_model_and_tokenizer(model_name)

    print('Loading data...')
    a1_data, a2_data, a3_data = load_a1_a2_a3_data()

    print('Simple prompting...')
    a1_prompts_answers, a2_prompts_answers, a3_prompts_answers = simple_process_a1_a2_a3_data(a1_data, a2_data, a3_data)

    print('Solution generating...')
    a1_prompts_answers_with_solutions, a2_prompts_answers_with_solutions, a3_prompts_answers_with_solutions = collect_simple_solutions(
        a1_prompts_answers, a2_prompts_answers, a3_prompts_answers, model, tokenizer, solution_nums=solution_nums, temperature=temperature
    )

    print('Writing file...')
    with open(os.path.join(path, f'a1_{type}_prompts_answers_with_solutions_{temperature}.jsonl'), 'w') as f:
        for data in a1_prompts_answers_with_solutions:
            f.write(json.dumps(data) + '\n')
    with open(os.path.join(path, f'a2_{type}_prompts_answers_with_solutions_{temperature}.jsonl'), 'w') as f:
        for data in a2_prompts_answers_with_solutions:
            f.write(json.dumps(data) + '\n')
    with open(os.path.join(path, f'a3_{type}_prompts_answers_with_solutions_{temperature}.jsonl'), 'w') as f:
        for data in a3_prompts_answers_with_solutions:
            f.write(json.dumps(data) + '\n')

