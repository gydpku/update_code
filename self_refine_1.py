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
from collections import Counter
#from torch.cuda.amp import autocast
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
    tokenizer = AutoTokenizer.from_pretrained(model_name,padding_side='left')
    return model, tokenizer
def random_sample(demos,num=4):
    import random
    sampled_demos=random.sample(demos,num)
    text='\nHere are some demonstration examples:'
    for example_id,example in enumerate(sampled_demos):
        text+='Example {0}: {1}'.format(example_id,example)
    return text
def generate_response(prompt, model, tokenizer,demos, solution_num=32, temperature=0.2, max_new_tokens=2048,num_iteration=1):
    start_time = time.time()
    
    #text=random_sample(demos,num=4) # Prepare messages for all solutions in one batch
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
        torch.cuda.empty_cache()
        
        # Decode responses in batch
        responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        solutions_iter=[]
        for iteration in range(num_iteration):
            feedback_messages = [
        [
            {"role": "system", "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."},
            {"role": "user", "content": "I will give you a question and a solution tried to answer it. Your job is to access this solution and give easy to follow feedback to improve it or solve its potential errors. If there are no improvement make, just say so. Question:\n"+prompt+'Solution:\n'+response}
        ]
        for response in responses
    ]
            texts = [
            tokenizer.apply_chat_template(
                feedback_message,
                tokenize=False,
                add_generation_prompt=True
            ) for feedback_message in feedback_messages
        ]

        # Tokenize all texts in one batch
            model_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)

            print('Generating responses in batch')
            generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature
                )
            torch.cuda.empty_cache()
            # Decode responses in batch
            feedback_responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            messages=[
        [
            {"role": "system", "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."},
            {"role": "user", "content": "Based on the feedback, improve the solution. Feedback:\n"+feedback+'\nSolution: '+response}
        ]
        for feedback,response in zip(feedback_responses,responses)
    ]
            texts = [
            tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True
            ) for message in messages
        ]

            model_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)

            print('Generating responses in batch')
            generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature
                )
            torch.cuda.empty_cache()
            responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            if iteration==0:
                solutions_iter.append(responses)
            if iteration==2:
                solutions_iter.append(responses)
            if iteration==4:
                solutions_iter.append(responses)
        print(iteration)
        end_time = time.time()
        print('Time taken:', end_time - start_time)
        return solutions_iter
    except Exception as e:
        print(f"Error occurred during generation: {e}")
        return []

    
    return responses

# Data processing and solution generation
def process_batch(data_dict,n=2,num_iteration=1):
    answers_with_solutions=[]
    for data in data_dict:
        prompt = data['Input']
        other_data_points = ['Input:'+d['Input']+'\nOutput:'+data['COT_Output'] for d in a1_prompts_answers if d != data]
        print('a1', data['Input'], data['Output'], data['COT_Output'], '\n')
        solutions=[]
        for _ in range(n):
            solution = generate_response(prompt, model, tokenizer,other_data_points, solution_num=solution_nums//n, temperature=temperature,num_iteration=1)
            concat_solutions=[]
            for S,s in zip(solutions,solution):
                S+=s
                concat_solutions.append(S)
            solutions=concat_solutions
        data['Solutions'] = solutions
        print(solutions)
        data['Temperature'] = temperature
        answers_with_solutions.append(data)
    return answers_with_solutions


def collect_simple_solutions(a1_prompts_answers, a2_prompts_answers, a3_prompts_answers, model, tokenizer, solution_nums=32, temperature=0.2,num_iteration=5,divide_num=4):
    a1_prompts_answers_with_solutions = process_batch(a1_prompts_answers,n=divide_num,num_iteration=num_iteration)
    for id,a1_prompts_answers_with_solution in enumerate(a1_prompts_answers_with_solutions):
        acc=calculate_accuracy(a1_prompts_answers_with_solution)
        print('a1_{0}_acc:{1}'.format(id,acc))
    #a1_acc=calculate_accuracy(a1_prompts_answers_with_solutions)   
    a2_prompts_answers_with_solutions = process_batch(a2_prompts_answers,n=divide_num,num_iteration=num_iteration)
    for id,a2_prompts_answers_with_solution in enumerate(a2_prompts_answers_with_solutions):
        acc=calculate_accuracy(a2_prompts_answers_with_solution)
        print('a2_{0}_acc:{1}'.format(id,acc))
    #a2_acc=calculate_accuracy(a2_prompts_answers_with_solutions)
    #print('a2',a2_acc)
    a3_prompts_answers_with_solutions = process_batch(a3_prompts_answers,n=divide_num,num_iteration=num_iteration)
    for id,a3_prompts_answers_with_solution in enumerate(a3_prompts_answers_with_solutions):
        acc=calculate_accuracy(a3_prompts_answers_with_solution)
        print('a3_{0}_acc:{1}'.format(id,acc))
    return a1_prompts_answers_with_solutions, a2_prompts_answers_with_solutions, a3_prompts_answers_with_solutions

# Main logic
if __name__ == "__main__":
    path = '/dccstor/obsidian_llm/yiduo'
    temperature = 0.7
    solution_nums = 32
    model_name = "/dccstor/obsidian_llm/yiduo/models--Qwen--QwQ-32B-Preview"
    type='in_context_random_1_self_refine'
    print("Initializing model and tokenizer...")
    model, tokenizer = initialize_model_and_tokenizer(model_name)

    print('Loading data...')
    a1_data, a2_data, a3_data = load_a1_a2_a3_data()
    import pdb
    print('Simple prompting...')
    a1_prompts_answers, a2_prompts_answers, a3_prompts_answers = simple_process_a1_a2_a3_data(a1_data, a2_data, a3_data)
    print('Solution generating...')
    a1_prompts_answers_with_solutions, a2_prompts_answers_with_solutions, a3_prompts_answers_with_solutions = collect_simple_solutions(
        a1_prompts_answers, a2_prompts_answers, a3_prompts_answers, model, tokenizer, solution_nums=solution_nums, temperature=temperature,num_iteration=5
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

