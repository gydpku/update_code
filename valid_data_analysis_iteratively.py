from batch_inference import batch_inference
from openai_call import query_azure_openai_chatgpt_chat
from generate_data import data_sample
import re
import pdb
from typing import List, Dict, Optional, Any
import re

def generate_data_prompts(data_group: List[tuple], task_instruction: str) -> List[str]:
    """Generate prompts for data characteristic analysis."""
    return [
        f"""
        Here is a task:
        {task_instruction}
        We have the following data and its correct solution:
        {data[0]}
        Your task is to find all challenging and difficult data characteristics of this data.
        We prefer to find the characteristics in task problem solving solution and reasoning process rather than specific knowledge or information.
        Directly output all challenging and difficult data characteristics of this data in reasoning and problem solving likes:
        characteristic:xxx (1~2 sentences for descripring the characteristic and its diffculty)
        ..."""
        for data in data_group
    ]

def generate_challenge_prompts(
    data_chars: List[str],
    data_group: List[tuple],
    task_instruction: str,
    previous_analysis: Optional[str] = None
) -> List[str]:
    """Generate prompts for challenge analysis."""
    if previous_analysis:
        return [
        f"""
        Given you the task data, its data characteristics, its correct solution and its wrong solution, 
        your task is to find the data characteristic/characteristics that causally lead to the wrong solution.
        We consider the following task and its data:
        Task: {task_instruction}\n
        Data query and its correct solution: {data[0]}\n
        Wrong solution: {data[1][0]}\n
        We also have its all challenging and difficult data characteristics:
        {data_char} 
        which characteristic/characteristics lead to this error?
        Do not consider the data characteristic that appears in the following list:
        {previous_analysis}
        Directly output the characteristic/characteristics. If you do not find the correct characteristic for the list, then you directly write the correct one.
        Your output format should follow the format of the original characteristics like:
        characteristic:xxx (its description sentence)
        ...
        """
        for data_char, data in zip(data_chars, data_group)
    ]
    return [
        f"""
        Given you the task data, its data characteristics, its correct solution and its wrong solution, 
        your task is to find the data characteristic/characteristics that causally lead to the wrong solution.
        We consider the following task and its data:
        Task: {task_instruction}\n
        Data query and its correct solution: {data[0]}\n
        Wrong solution: {data[1][0]}\n
        We also have its all challenging and difficult data characteristics:
        {data_char} \n
        which characteristic/characteristics lead to this error?
        Directly output the characteristic/characteristics. If you do not find the correct characteristic for the list, then you directly write the correct one.
        Your output format should follow the format of the original characteristics like:
        characteristic:xxx (its description sentence)
        ...
        """
        for data_char, data in zip(data_chars, data_group)
    ]

def generate_pattern_prompt(
    case_analysis: List[str],
    task_instruction: str,
    previous_analysis: Optional[str] = None
) -> str:
    """Generate prompt for pattern analysis."""
    import pdb
#    pdb.set_trace()
    number_prompt="""Here are some case analysis of the challenging characteristic of the task data.
    You need to decide the number of main data characteristic patterns to cover all challenge and diffculty among these cases.
    Each pattern should be different from each other (exclusive) and should cover cases as more as possible.
    Directly output the minimum number of patterns to cover all cases, following with your answer.
    """
 #   pdb.set_trace()
    case_text=''
    for case_id, case in enumerate(case_analysis):
        case_text+=f"Case {case_id}:{case}\n"
    number_prompt+=f"cases:{case_text}"
  #  pdb.set_trace()
    number_analysis=query_azure_openai_chatgpt_chat(number_prompt, temperature=0.0)
    import re
    extract_number=re.findall(r'\d+',number_analysis)
    num_pattern=min(int(extract_number[0]),3)
    import pdb
    prompt = f"""
        Here are some case analysis of the challenging characteristic of the task data.
        You need to induce {num_pattern} main data characteristic patterns from these case analysis in general.
        The pattern should capture the challenge and diffculty among these cases.
        Each pattern should be different from each other (exclusive).
        And these patterns should cover all cases.
        """
    
    #data driven
    # Add cases
    for case_id, case in enumerate(case_analysis):
        prompt += f"Case {case_id}:{case}\n"
    
    # Add previous analysis if available
#    pdb.set_trace()
    refine_prompt="""These previous characteristic can not capture the characteristic of these cases fully. You can analyze and direcly output features of these cases that are important but ignored by previous characteristics directly.
    previous characteristic:{0} cases:{1}""".format(previous_analysis,case_text)
    #analysis=query_azure_openai_chatgpt_chat(refine_prompt)
    
    if previous_analysis:
        analysis=query_azure_openai_chatgpt_chat(refine_prompt)
        prompt += f"""Here are some considered data characteristic:
        {previous_analysis}. 
        You should generate new characteristic patterns that are different from them.
        You can refer to these new important features {analysis}
        """
#    pdb.set_trace()
    # Add output format instructions
    prompt += """Directly output induced characteristic patterns and the id of their cases in the following format: 
        #characteristic 1: (1~2 description sentences)
        cases number id : xxx,xxx,xxx (xxx are the cases id of cases that belong to this characteristic)
        #characteristic 2: (1~2 description sentences)
         cases number id : xxx,xxx,xxx (xxx is the case id of cases that belong to this characteristic)
        ...
        # is necessary to split the characteristic and the cases number id by a new line
        """
    
    return prompt

def characteristic_analysis_to_dict(
    characteristic_analysis: str,
    data_group: List[tuple],
    is_num: bool = False
) -> Dict[str, List[tuple]]:
    """Convert characteristic analysis to dictionary."""
    characteristic_dict = {}
    lines = characteristic_analysis.strip().split('#')
#    pdb.set_trace()
    for line in lines:
        if 'characteristic' in line:
            try:
                characteristic, case_num = line.split('cases number id')
                numbers = [int(num) for num in re.findall(r'\d+', case_num)]
                characteristic = characteristic.split(':')[1].strip()
                
                characteristic_dict[characteristic] = [
                    data_group[i] if not is_num else i for i in numbers
                    if i < len(data_group)
                ]
                
            except Exception as e:
                print(f"Error processing characteristic line: {str(e)}")
                continue
                
    return characteristic_dict

def extract_number(text):
    """
    Extracts the first standalone number from a given string.
    
    Parameters:
    text (str): The input string to search for a number.
    
    Returns:
    str: The extracted number as a string, or None if no number is found.
    """
    pattern = r'\b(\d+)\b'  # Matches standalone numbers
    match = re.search(pattern, text)
    return int(match.group(1)) if match else None
def knowledge_based_or_task_based_identify(case_analysis: List[str],task_instruction: str) -> List[str]:
    return [f""" Your task is to identify whether the challenge of the following case is knowledge-based or task-based.
            Knowledge-based challenge means the case mainly needs specific knowledge or information to solve it.
            Task-based challenge means the case mainly needs task-relation solultion skills and ability to solve it.
            Case: {case}
            Task: {task_instruction}
    Directly output your answer in the following format:
    'knowledge-based' or 'task-based
    """ for case in case_analysis]
import random
def process_case_analysis(challenge_prompts,temperature=0.2):
    """Extract and process case analysis from challenge analysis."""
    challenge_analysis = batch_inference(challenge_prompts, temperature=temperature)
 #   pdb.set_trace()
    c_a=[' '.join(analysis.lower().split('characteristic:')[1::2]) for analysis in challenge_analysis]
    random.shuffle(c_a)
    return c_a #random.shuffle([' '.join(analysis.lower().split('characteristic:')[1::2]) for analysis in challenge_analysis])

def get_initial_characteristics(case_analysis, task_instruction, previous_analysis_text,data_group,temperature=0.2):
    """Generate initial characteristic analysis."""
    pattern_prompt = generate_pattern_prompt(case_analysis, task_instruction, previous_analysis=previous_analysis_text)
    characteristic_analysis = query_azure_openai_chatgpt_chat(pattern_prompt, temperature=temperature)
    return characteristic_analysis_to_dict(characteristic_analysis, data_group, is_num=True)

def check_and_revise_characteristic(key, value, previous_analysis_text, previous_analysis, case_analysis):
    """Check if characteristic exists in previous analysis and revise if needed."""
    prompt = f"Does this characteristic: {key} appears in the following characteristic pattern: {previous_analysis_text}?\nDirectly answer yes+characteristic id or no."
    res = query_azure_openai_chatgpt_chat(prompt, temperature=0.0)
    
    if "yes" not in res.lower():
        return None, key, value
    
    prev_id = extract_number(res)
    prev_char = previous_analysis[prev_id]
    
    # Generate revision prompt
    prompt = f"\nCurrent characteristic pattern: {key} is too vague and pretty similar to previous one: {prev_char}."
    prompt += "\nYou should make it different from the previous one by considering more distinguishable details from these cases."
    
    for id, case_id in enumerate(value):
        prompt += f"Case {id}:{case_analysis[case_id]}\n"
    prompt += "Direcly output your new characteristic pattern. You should still follow this pattern format: characteristic: (1~2 description sentences)+ (1~2 detailed description) sub-characteristics including: (2~3 distinguishable ones)."
    
    characteristic = query_azure_openai_chatgpt_chat(prompt, temperature=0.0)
    characteristic = characteristic.replace('characteristic:', '').replace('Characteristic:', '').strip()
    
    print('Yes', '\n', 'new one:', characteristic, '\n', 'old one:', key, '\n', 'previous one:', prev_char)
    return prev_id, characteristic, value

def merge_characteristics(records_prev, characteristic_dict_revised,temperature=0.0):
    """Merge related characteristics."""
    characteristic_dict = {}
    
    for prev_id, characteristics in records_prev.items():
        if len(characteristics) > 1:
            char_text = '\n'.join(f'{i}:{char}' for i, char in enumerate(characteristics))
            prompt = f"You should merge the following characteristics into one: {char_text}.\nDirectly output the merged one."
            
            merged_char = query_azure_openai_chatgpt_chat(prompt, temperature=0.0)
            merged_char = merged_char.replace('characteristic:', '').replace('Characteristic:', '').strip()
            print('merged one:', merged_char)
            
            characteristic_dict[merged_char] = []
            for char in characteristics:
                characteristic_dict[merged_char].extend(characteristic_dict_revised[char])
        else:
            char = characteristics[0]
            characteristic_dict[char] = characteristic_dict_revised[char]
    
    return characteristic_dict

def finalize_characteristics(characteristic_dict_new, data_group):
    """Create final characteristic dictionary with data group mapping."""
    return {
        key: [data_group[i] for i in value if i < len(data_group)]
        for key, value in characteristic_dict_new.items()
    }

# Main execution
def process_characteristics(challenge_prompts, task_instruction, previous_analysis_text, previous_analysis, data_group,temperature=0.2):
    """Main function to process and analyze characteristics."""
    case_analysis = process_case_analysis(challenge_prompts,temperature=temperature)
#    pdb.set_trace()
    characteristic_dict = get_initial_characteristics(case_analysis, task_instruction, previous_analysis_text,data_group,temperature=temperature)
#    pdb.set_trace()
    characteristic_dict_revised = {}
    characteristic_dict_new = {}
    records_prev = {}
    
    # Process each characteristic
    for key, value in characteristic_dict.items():
        prev_id, new_char, new_value = check_and_revise_characteristic(
            key, value, previous_analysis_text, previous_analysis, case_analysis
        )
        
        if prev_id is None:
            characteristic_dict_new[new_char] = new_value
        else:
            characteristic_dict_revised[new_char] = new_value
            if prev_id not in records_prev:
                records_prev[prev_id] = [new_char]
            else:
                records_prev[prev_id].append(new_char)
 #   pdb.set_trace()
    # Merge and finalize characteristics
    return finalize_characteristics(characteristic_dict_new,data_group),finalize_characteristics(merge_characteristics(records_prev, characteristic_dict_revised,temperature=temperature), data_group)
def number_first(challenge_prompts,task_instruction,previous_analysis_text,previous_analysis,data_group,iteration=5,graident_number=3):
    characteristic_dict={}
    for _ in range(iteration):
        characteristic_dict_new,characteristic_dict_revised=process_characteristics(challenge_prompts, task_instruction, previous_analysis_text, previous_analysis, data_group,temperature=0.2)
        characteristic_dict.update(characteristic_dict_new)
        characteristic_dict.update(characteristic_dict_revised)
    characteristic_dict_list=[]
    for key,value in characteristic_dict.items():
        characteristic_dict_list.append((key,value,len(value)))
    characteristic_dict_list.sort(key=lambda x:x[2],reverse=True)
    gradients={}
    for id in range(graident_number):
        gradients[characteristic_dict_list[id][0]]=characteristic_dict_list[id][1]
    return gradients
def new_first(challenge_prompts,task_instruction,previous_analysis_text,previous_analysis,data_group,iteration=5,graident_number=3):
    characteristic_dict_list=[]
    for _ in range(iteration):
        characteristic_dict_new,characteristic_dict_revised=process_characteristics(challenge_prompts, task_instruction, previous_analysis_text, previous_analysis, data_group,temperature=0.2)
        for key,value in characteristic_dict_new.items():
            characteristic_dict_list.append((key,value,1,len(value)))
    #    for key,value in characteristic_dict_revised.items():
     #       characteristic_dict_list.append((key,value,0,len(value)))
    
    characteristic_dict_list.sort(key=lambda x:(x[2],x[3]),reverse=True)
    gradients={}
    for id in range(min(graident_number,len(characteristic_dict_list))):
        gradients[characteristic_dict_list[id][0]]=characteristic_dict_list[id][1]
    return gradients
def random_select(challenge_prompts,task_instruction,previous_analysis_text,previous_analysis,data_group,iteration=5,graident_number=3):
    characteristic_dict_list=[]
    for _ in range(iteration):
        characteristic_dict_new,characteristic_dict_revised=process_characteristics(challenge_prompts, task_instruction, previous_analysis_text, previous_analysis, data_group,temperature=0.2)
  #      pdb.set_trace()
        for key,value in characteristic_dict_new.items():
            characteristic_dict_list.append((key,value,1,len(value)))
        for key,value in characteristic_dict_revised.items():
            characteristic_dict_list.append((key,value,0,len(value)))
    
    random.shuffle(characteristic_dict_list)
    gradients={}
    for id in range(graident_number):
        gradients[characteristic_dict_list[id][0]]=characteristic_dict_list[id][1]
    return gradients 
def data_type_analysis(
    data_group: List[tuple],
    task_instruction: str,
    previous_analysis: List[str] = None
) -> Dict[str, List[tuple]]:
    """
    Analyze data characteristics and patterns.
    
    Args:
        data_group: List of data tuples
        task_instruction: Task instruction string
        previous_analysis: Optional previous analysis string
    
    Returns:
        Dictionary mapping characteristics to data groups
    """
    try:
        # Generate and process data characteristics
        data_prompts = generate_data_prompts(data_group, task_instruction)
        data_chars = batch_inference(data_prompts, temperature=0.0)
#        pdb.set_trace()
        previous_analysis_text=''
        if previous_analysis:
            previous_analysis_text=''
            for id, pre_analysis in enumerate(previous_analysis):
                previous_analysis_text += 'characteristic {0}:{1}\n'.format(id,pre_analysis)
            challenge_prompts = generate_challenge_prompts(data_chars, data_group, task_instruction, previous_analysis_text)
        else:
            challenge_prompts = generate_challenge_prompts(data_chars, data_group, task_instruction)
 #       pdb.set_trace()
        #characteristics=new_first(challenge_prompts,task_instruction,previous_analysis_text,previous_analysis,data_group,iteration=5,graident_number=3)
#        pdb.set_trace()
       # return characteristics   
        challenge_analysis = batch_inference(challenge_prompts, temperature=0.0)
        #pdb.set_trace()
        '''
        knowledge_based_or_task_based_prompts = knowledge_based_or_task_based_identify(challenge_analysis,task_instruction)
        knowledge_based_or_task_based_analysis = batch_inference(knowledge_based_or_task_based_prompts, temperature=0.0)
        knowledge_number=sum([1 for analysis in knowledge_based_or_task_based_analysis if 'knowledge-based' in analysis.lower()])
        task_number=sum([1 for analysis in knowledge_based_or_task_based_analysis if 'task-based' in analysis.lower()])
        print('knowledge-based:',knowledge_number,knowledge_number/len(knowledge_based_or_task_based_analysis),'task-based:',task_number,task_number/len(knowledge_based_or_task_based_analysis))
#        pdb.set_trace()

        '''
        # Process case analysis
        case_analysis = [' '.join(analysis.lower().split('characteristic:')[1::2]) for analysis in challenge_analysis]
  #      pdb.set_trace()        
        # Generate pattern analysis
        pattern_prompt = generate_pattern_prompt(case_analysis, task_instruction, previous_analysis=previous_analysis)
        pdb.set_trace()
        characteristic_analysis = query_azure_openai_chatgpt_chat(pattern_prompt, temperature=0.0)
#        pdb.set_trace()
        characteristic_dict = characteristic_analysis_to_dict(characteristic_analysis, data_group, is_num=False)
        #pdb.set_trace()
        return characteristic_dict
        '''
        #pdb.set_trace()
        # Convert to dictionary
        characteristic_dict_revised={}
        characteristic_dict_new={}
        records_prev={}
        for key, value in characteristic_dict.items():
#            pdb.set_trace()
            prompt="Does this characteristic: {0} appears in the following characteristic pattern: {1}?\nDirectly answer yes+characteristic id or no.".format(key, previous_analysis_text)
            res = query_azure_openai_chatgpt_chat(prompt, temperature=0.0)
            if "yes" in res.lower():
                prev_id = extract_number(res)
                prev_char=previous_analysis[prev_id]
                
                prompt="\nCurrent characteristic pattern: {0} is too vague and pretty similar to previous one: {1}.\nYou should make it different from the previous one by considering more distinguishable details from these cases.".format(key,prev_char)
                for id, case_id in enumerate(value):
                    prompt += "Case {0}:{1}\n".format(id,case_analysis[case_id])
                prompt += "Direcly output your new characteristic pattern. You should still follow this pattern format: characteristic: (1~2 description sentences)+ sub-characteristics including: (2~3 distinguishable ones)."
                characteristic = query_azure_openai_chatgpt_chat(prompt, temperature=0.0)
                characteristic=characteristic.replace('characteristic:','').replace('Characteristic:','').strip()
                print('Yes','\n','new one:',characteristic,'\n','old one:',key,'\n','previous one:',prev_char)
                characteristic_dict_revised[characteristic] = value
                if prev_id not in records_prev:
                    records_prev[prev_id]=[characteristic]
                else:
                    records_prev[prev_id].append(characteristic)
            else:
                characteristic_dict_new[key] = value
#        pdb.set_trace()
        # merge the ones
        for prev_id in records_prev:
            if len(records_prev[prev_id])>1:
                char_text=''
                for char_id,char in enumerate(records_prev[prev_id]):
                    char_text+='{0}:{1}\n'.format(char_id,char)
                prompt="You should merge the following characteristics into one: {0}.\nDirectly output the merged one.".format(char_text)
                merged_char=query_azure_openai_chatgpt_chat(prompt, temperature=0.0)
                merged_char=merged_char.replace('characteristic:','').replace('Characteristic:','').strip()
                print('merged one:',merged_char)
                characteristic_dict_new[merged_char] = []
                for char in records_prev[prev_id]:
                    characteristic_dict_new[merged_char].extend(characteristic_dict_revised[char])
            else:
                characteristic_dict_new[records_prev[prev_id][0]] = characteristic_dict_revised[records_prev[prev_id][0]]
 #       pdb.set_trace()
        characteristic_dict={}
        for key, value in characteristic_dict_new.items():
            characteristic_dict[key] = [
                    data_group[i] for i in value
                    if i < len(data_group)
                ]
  #      pdb.set_trace()
        '''
        characteristic_dict={}
        for key, value in characteristic_dict_number.items():
            characteristic_dict[key] = [
                    data_group[i] for i in value
                    if i < len(data_group)
                ]       
        return characteristic_dict
        
    except Exception as e:
        print(f"Error in data type analysis: {str(e)}")
        return {}
def remove_characteristics(characteristic_dict: Dict[str, Any], del_keys: List[str]) -> Dict[str, Any]:
    """
    Safely remove keys from characteristic dictionary.
    
    Args:
        characteristic_dict: Original dictionary
        del_keys: List of keys to delete
        
    Returns:
        Updated dictionary with keys removed
    """
    # Create a new dictionary excluding the keys to delete
    return {
        key: value 
        for key, value in characteristic_dict.items()
        if key not in del_keys
    }
