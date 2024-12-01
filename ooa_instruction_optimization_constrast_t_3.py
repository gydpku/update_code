from valid_data_analysis_iteratively import data_type_analysis
from batch_inference import batch_inference
from openai_call import query_azure_openai_chatgpt_chat
from generate_data import data_sample,data_sample_pattern
import torch
import re
import random
import pdb
import re

def extract_float(text):
    # Regular expression to match a float starting with "0."
    try:
        float_number = re.findall(r"\b0\.\d+\b", text)
    except:
        print('Response error!')
        return 0
    # Return the first match as a float, or 0 if no match is found
    return float(float_number[0]) if float_number else 0
def generation_instruction_generation(characteristic,examples, task_instruction,previous_instruction=None,prompt_num=5,temperature=0.7):
    prompt="""
    Here is a task:
    {0}
    We lack task data that has the following characteristic:
    {1}
    For example:
    {2}
    Your task is to provide one data generation instruction for generate task data that has the given characteristic.
    Directly output the instruction in the following format:
    Instruction: xxx
    Goal: xxx (the purpose of the data)
    Content form: xxx (the format, structure, and length of the data you need)
    Specific Constraints: xxx (necessary conditions that the data must satisfy)
    """.format(task_instruction, characteristic,random.choice(examples))
    if previous_instruction:
        prompt+="You can refer to the previous instruction and generate a better instruction: {0}".format(previous_instruction)
    prompts=[prompt]*prompt_num
    return batch_inference(prompts,temperature=temperature)
def discrimination_instruction_generation(characteristic,examples, task_instruction,previous_instruction=None,prompt_num=5,temperature=0.7):
    prompt="""
    Here is a task:
    {0}
    We only need task data that has the following characteristic:
    {1}
    For example:
    {2}
    Your task is to provide one discriminative instruction for distinguishing data that has the given characteristic from other data.
    Directly output the discriminative instruction in the following format:
    Instruction: xxx
    """.format(task_instruction, characteristic,random.choice(examples))
    if previous_instruction:
        prompt+="You can refer to the previous instruction and generate a better instruction: {0}".format(previous_instruction)
    prompts=[prompt]*prompt_num
    return batch_inference(prompts,temperature=temperature)

def evaluate_instructions(data, data_list, characteristic_or_instruction, is_discriminative=False):
    """Helper function to evaluate instructions based on A/B comparisons"""
    prompts = []
    
    for sample, demo_sample in zip(data, data_list):
        prompt_template = """Your task is to calculate the similarity score of two examples only based on this data characteristic {0}.
A higher score means more similarity. 
    A:{1}\n
    B:{2}
    Directly and only output the score in float form (0~1)."""
        
        prompts.append(prompt_template.format(characteristic_or_instruction, sample, demo_sample))
        #prompts_2.append(prompt_template.format(characteristic_or_instruction, demo_sample, sample))
    
    results = batch_inference(prompts, temperature=0.0)
#    pdb.set_trace()
    # Extract just 'A' or 'B' from responses
    results = [extract_float(r) for r in results] #'True' if r and 'True' in r else 'False' if r else 'N/A' for r in results]
    #pdb.set_trace() 
    return sum([1-result for result in results]) if is_discriminative else sum(results)
#    results_1 = ['A' if 'A' in r else 'B' for r in results_1]
 #   results_2 = ['A' if 'A' in r else 'B' for r in results_2]
    
    # For discriminative instructions, we expect B/A pattern
    # For generative instructions, we expect A/B pattern
    #expected_pattern = 'False' if is_discriminative else 'True'
    #pdb.set_trace()
    #return sum(1 for r in results if r == expected_pattern)
def generate_data(char_instruction, reward_prompt, domain, data_num, store_name, task_name):
    """Helper function to generate data and store it"""
    data = []
    while len(data) < data_num:
        new_data = data_sample_pattern(
        char_instruction, domain, data_num-len(data),
        store_name + str(random.randint(10**14, 10**15 - 1)),
        reward_prompt, task_name,
        pattern=True, neg_sample=False
    )
    # Filter out None values directly when extending data
        data.extend([item for item in new_data if item is not None])
    return data[:data_num] #pdb.set_trace()
    #data=data_sample_pattern(char_instruction,domain,data_num,store_name+str(random.randint(10**14,10**15-1)),reward_prompt,task_name,pattern=True,neg_sample=False)
#    pdb.set_trace() 
    #return data
def process_characteristics_and_instructions(
    characteristic_dict: dict,
    ground_data: list,
    task_instruction: str,
    reward_prompt: str,
    domain: str,
    data_num: int,
    store_name: str,
    task_name: str,
    prompt_num: int = 50  # Added prompt_num parameter with default value
) -> dict:
    """
    Process characteristics and generate optimized instructions with their corresponding data.
    
    Args:
        characteristic_dict: Dictionary of characteristics
        ground_data: List of ground truth data
        task_instruction: Task instruction string
        reward_prompt: Reward prompt string
        domain: Domain string
        data_num: Number of data samples to generate
        store_name: Name for storing data
        task_name: Name of the task
        prompt_num: Number of prompts to generate for each characteristic (default: 50)
    
    Returns:
        dict: Dictionary mapping characteristics to (best_instruction, ground_data, generated_data)
    """
    # Step 1: Generate instructions and data for each characteristic
    char_data_instructions = {}
    for characteristic in characteristic_dict:
        char_data_instructions[characteristic] = {}
        char_instructions = generation_instruction_generation(
            characteristic=characteristic,
            examples=ground_data,
            task_instruction=task_instruction,
            prompt_num=prompt_num  # Using the prompt_num parameter
        )
        # Generate data for each instruction
        for char_instruction in char_instructions:
            data = generate_data(
                char_instruction=char_instruction,
                reward_prompt=reward_prompt,
                domain=domain,
                data_num=data_num,
                store_name=store_name,
                task_name=task_name
            )
            char_data_instructions[characteristic][char_instruction] = data
    
    # Step 2: Evaluate instructions and select the best ones
    ooa_instructions_data = {}
    for characteristic in characteristic_dict:
        # Prepare evaluation data
        ground_data = [data[0] for data in characteristic_dict[characteristic]]
        other_key_data = sample_other_key_data(
            data_dict=char_data_instructions,
            obj_key=characteristic,
            data_num=data_num
        )
        
        # Score each instruction
        instructions_scores = []
        for char_instruction in char_data_instructions[characteristic]:
            generated_data = char_data_instructions[characteristic][char_instruction]
            positive_score = evaluate_instructions(
                data=generated_data,
                data_list=ground_data,
                characteristic_or_instruction=characteristic,
                is_discriminative=False
            )
            negative_score = evaluate_instructions(
                data=generated_data,
                data_list=other_key_data,
                characteristic_or_instruction=characteristic,
                is_discriminative=False
            )
            score = positive_score / (positive_score + negative_score)
            instructions_scores.append((char_instruction, score))
        
        # Select best instruction
        instructions_scores.sort(key=lambda x: x[1], reverse=True)
        best_instruction = instructions_scores[0][0]
        ooa_instructions_data[characteristic] = (
            best_instruction,
            ground_data,
            char_data_instructions[characteristic][best_instruction]
        )
    
    return ooa_instructions_data
def sample_other_key_data(data_dict,obj_key,data_num):
    """Helper function to sample other key data"""
    other_key_data=[]
    for key in data_dict:
        if key!=obj_key:
            for instruction in data_dict[key]:
                other_key_data.extend(data_dict[key][instruction])
    random_sample_data=random.sample(other_key_data,data_num)
    return random_sample_data
def generate_generative_instructions(characteristic_dict,task_instruction,reward_prompt,domain,data_num,store_name,task_name,prompt_num=50,previous_instructions=None):
            char_data_instructions={}
            for characteristic in characteristic_dict:
                char_data_instructions[characteristic]={}
                ground_data=[data[0] for data in characteristic_dict[characteristic]]
                if previous_instructions:
                    char_instructions = generation_instruction_generation(characteristic,ground_data, task_instruction,previous_instructions[characteristic],prompt_num=prompt_num)
                else:
                    char_instructions = generation_instruction_generation(characteristic,ground_data, task_instruction,prompt_num=prompt_num)
                for char_instruction in char_instructions:
                    data = generate_data(char_instruction,reward_prompt, domain, data_num, store_name, task_name)
                    char_data_instructions[characteristic][char_instruction]=data
#                    pdb.set_trace()
            return char_data_instructions
def optimize_generative_instructions(char_data_instructions,characteristic_dict,data_num,discriminative_instructions):     
        ooa_instructions_data={}
        for characteristic in characteristic_dict:
            instructions_scores=[]
            ground_data=[data[0] for data in characteristic_dict[characteristic]]
            other_key_data=sample_other_key_data(char_data_instructions,characteristic,data_num)
            valid_ground_data=[]
            for _ in range(data_num):
                valid_ground_data.append(random.choice(ground_data))
            for char_instruction in char_data_instructions[characteristic]:
                positve_score=evaluate_instructions(char_data_instructions[characteristic][char_instruction],valid_ground_data,discriminative_instructions[characteristic],is_discriminative=False)
                negative_score=evaluate_instructions(char_data_instructions[characteristic][char_instruction],other_key_data,discriminative_instructions[characteristic],is_discriminative=False)
#                pdb.set_trace()
                instructions_scores.append((char_instruction,positve_score/(positve_score+negative_score)))
            instructions_scores.sort(key=lambda x:x[1],reverse=True)
            best_instruction=instructions_scores[0][0]
#            pdb.set_trace()
            ooa_instructions_data[characteristic]=(best_instruction,ground_data,char_data_instructions[characteristic][best_instruction])
        return ooa_instructions_data
def generate_and_optimize_discriminative_instructions(ooa_instructions_data,characteristic_dict,task_instruction):
        discriminative_instructions={}
        for characteristic in characteristic_dict:
            ground_data=[data[0] for data in characteristic_dict[characteristic]]
            char_discriminative_instructions = discrimination_instruction_generation(characteristic,ground_data, task_instruction)
            scores = []
            
            for char_discriminative_instruction in char_discriminative_instructions:
                scores.append(evaluate_instructions(ooa_instructions_data[characteristic][2], ooa_instructions_data[characteristic][1], char_discriminative_instruction, is_discriminative=True))
            best_discriminative_instruction = char_discriminative_instructions[scores.index(max(scores))]
            discriminative_instructions[characteristic]=best_discriminative_instruction
        return discriminative_instructions
def run_ooa_instruction_optimization(ooa_data,task_instruction,reward_prompt,task_name,store_name,domain,data_num=20,previous_gradients=None):
#    pdb.set_trace() 
    try:
   
        characteristic_dict=torch.load(store_name+"_characteristic_dict.pt")
 #       pdb.set_trace()
#        characteristic_dict=data_type_analysis(ooa_data,task_instruction,previous_analysis=previous_gradients)
    except: 
#        aaa=torch.load('ooa_constrast_3_r_characteristic_dict.pt')
 #       keys_text=[key for key in aaa.keys()]
        characteristic_dict=data_type_analysis(ooa_data,task_instruction,previous_analysis=previous_gradients) #,previous_analysis=keys_text)
#        pdb.set_trace()
        torch.save(characteristic_dict, store_name+"_characteristic_dict.pt")
 #   store_name+='refine' #    pdb.set_trace()
    try:
        ooa_instructions_data=torch.load(store_name+"_ooa_instructions_data_t_3.pt")
    except:
        ooa_instructions_data={}
        char_data_instructions={}
        discriminative_instructions={}
        for characteristic in characteristic_dict:
            discriminative_instructions[characteristic]=characteristic
        previous_instructions=None
        for iteration in range(1):      
        # generate data and instructions candidates for each characteristic
            #char_data_instructions=torch.load('char_data_instructions.pt')
            
            if previous_instructions:
                char_data_instructions=generate_generative_instructions(characteristic_dict,task_instruction,reward_prompt,domain,data_num,store_name,task_name,prompt_num=50,previous_instructions=previous_instructions)
            else:
                char_data_instructions=generate_generative_instructions(characteristic_dict,task_instruction,reward_prompt,domain,data_num,store_name,task_name,prompt_num=50)
            torch.save(char_data_instructions,store_name+'char_data_instructions_t_3_{0}.pt'.format(iteration))
#            pdb.set_trace()
            #print(iteration)
            ooa_instructions_data=optimize_generative_instructions(char_data_instructions,characteristic_dict,data_num,discriminative_instructions)
            previous_instructions={}
            for characteristic in characteristic_dict:
                previous_instructions[characteristic]=ooa_instructions_data[characteristic][0]
        # Step 2: Generate discriminative instructions and evaluate them
            discriminative_instructions=generate_and_optimize_discriminative_instructions(ooa_instructions_data,characteristic_dict,task_instruction)
            torch.save(char_data_instructions,store_name+'ooa_instructions_data_t_3_{0}.pt'.format(iteration))
            torch.save(discriminative_instructions,store_name+'discriminative_instructions_t_3_{0}.pt'.format(iteration))
#        pdb.set_trace()
            print(iteration,previous_instructions) #pdb.set_trace() #ooa_instructions_data[characteristic]=(best_discriminative_instruction,ooa_instructions_data[characteristic][1],ooa_instructions_data[characteristic][2])
        data_num=50
        char_data_instructions=generate_generative_instructions(characteristic_dict,task_instruction,reward_prompt,domain,data_num,store_name,task_name,prompt_num=5,previous_instructions=previous_instructions)
        #pdb.set_trace()    
        ooa_instructions_data=optimize_generative_instructions(char_data_instructions,characteristic_dict,data_num,discriminative_instructions)
        torch.save(ooa_instructions_data, store_name+"_ooa_instructions_data_t_3.pt")
 #       pdb.set_trace() 
        for char in ooa_instructions_data:
            char_instruction=ooa_instructions_data[char][0]
            ooa_instructions_data[char][2].extend(generate_data(char_instruction, '', domain, 450, store_name, task_name))
        torch.save(ooa_instructions_data, store_name+"_ooa_instructions_data_t_3.pt")
    return ooa_instructions_data



