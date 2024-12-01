import pdb
from statistics import mode
from utils import extract_answer_prediction_nli
from batch_inference import batch_inference
from reward_prompt_selection import best_example_selection
from func_timeout import func_timeout, FunctionTimedOut
import signal
from collect_data import prompt_domain_find_wikipassage
from eval import evaluate_score,evaluate_score_reason,evaluate_score_self
#from generate_data import extract_example
#from generate_data import multiple_examples_extraction
import random
import json
from datasets import Dataset,DatasetDict
from datasets import load_dataset
from openai_call import query_azure_openai_chatgpt_chat ,query_azure_openai_chatgpt_chat_2
import torch
import torch
import numpy as np
import re
import random
seed=2024
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(seed)
META_REVISION_PROMPT="""As a DatasetGenerator, your task is to re-write a given example into a more complex version to make those famous AI systems (e.g., ChatGPT and GPT4) a bit harder to handle. But the rewritten example must be reasonable and must be understood and responded by humans.
Your rewriting cannot omit the non-text parts such as the table and code in the given example. Also, please do not omit the input and output structure in the given example. You SHOULD complicate the given example using the following method: """
META_PROMPT = """
As a DatasetGenerator, your task is to generate one new example (`input` and `output`) based on the [new instruction], [reference passage], and [few-shot examples]. Please provide a JSON dictionary response that includes the new `input` and its corresponding `output`. Use the `input` and `output` keys in the dictionary.
Try you best to ensure that the input and output you generate are distinct from the provided examples while maintaining a diverse, detailed, precise, comprehensive, and high-quality response.
"""
META_PROMPT_simple = """
As a DatasetGenerator, your task is to generate one new examples (`input` and `output`) based on the [new instruction] and [few-shot examples]. Please provide a JSON dictionary response that includes the n$
Try you best to ensure that the input and output you generate are distinct from the provided examples while maintaining a diverse, detailed, precise, comprehensive, and high-quality response.
"""

def example_check(new_example,obj_passage=None):
        new_example=new_example[new_example.find('{'):new_example.rfind('}')+1]
        try:
            new_example=extract_examples(new_example)[0]
        except:
            return
        #pdb.set_trace()
        ''' 
        try:
            new_example['input']="[INST] "+"""Here is a database schema:{0} """.format(obj_passage[0][1])+'None'+new_example['input']
        except:
            return
        '''
        return new_example
def aug_few_shot_prompt_pattern_generate(high_quality_examples,task_instruction,obj_passage,reward_prompt,operation_instruction=None,neg_sample=True,revise_instruction=None):
#aug_few_shot_prompt_pattern_generate(r_examples,task_instruction,sampled_objects,reward_prompt,neg_sample=neg_sample)    
    template=META_PROMPT
    template+='You must consider the task instruction (task knowledge), provided examples (format), and the passage (domain knowledge) to generate your training data.'
    template+=""" Here is the task instruction:{0}\n""".format(task_instruction)
    template+=" Here is some demonstration examples. You should follow the input and output pattern of examples strictly to generate data!!!" #and the input and output's format pattern in general:{1}. The output includes its soluti$
    for id in range(len(high_quality_examples)):
        template+='Demo Example {0}: {1}'.format(id,high_quality_examples[id])
    template+="Here is some related objects or passages that you can refer to." #+operation_instruction[0]  #"For example, you can utilize its information to construct the new training data." # You can fol$
    template+="Related Objects or Passages:{0}".format(obj_passage[0][:min(2048,len(obj_passage[0]))])
    new_examples=[]
    template+="Before generating the new example, ensure that you strictly adhere to the rules mentioned in the [Requirement] and follow the format of the [high-quality examples]. Think twice before generating a new example. New example (in JSON):"
    sample_num=5 if neg_sample else 1
    return template
    #new_example=query_azure_openai_chatgpt_chat(template,temperature=0.7,n=sample_num)
    #best_example=new_example #random.choice(new_examples)
    #pdb.set_trace()
    #return [example_check(best_example,obj_passage)],example_check(best_example,obj_passage)
def aug_few_shot_prompt_generate(high_quality_examples,task_instruction,obj_passage,reward_prompt,operation_instruction=None,neg_sample=True,revise_instruction=None):
    
    template=META_PROMPT
    template+='You must consider the task instruction (task knowledge), provided examples (format), and the passage (domain knowledge) to generate your training data.'
    template+=""" Here is the task instruction:{0}\n""".format(task_instruction)
    template+=" Here is some demonstration examples. You should follow the examples strictly to generate data!!!" #and the input and output's format pattern in general:{1}. The output includes its solution steps and the final result. You should follow the pattern strictly to generate data!!!".format(high_quality_examples,pattern)
    for id in range(len(high_quality_examples)):
        template+='Demo Example {0}: {1}'.format(id,high_quality_examples[id])
    template+="Here is some related objects or passages that you can refer to."+operation_instruction[0]  #"For example, you can utilize its information to construct the new training data." # You can follow this procedure to construct the training sample:{0}.".format(procedure) # For example, you can rephrase it or utilize its domain knowledge in text to construct the new training data."
    template+="Related Objects or Passages:{0}".format(obj_passage[0][:min(2048,len(obj_passage[0]))])
#    api_docs=""
    print(operation_instruction[0],'\n')

    new_examples=[]
    template+="Before generating the new example, ensure that you strictly adhere to the rules mentioned in the [Requirement] and follow the format of the [high-quality examples]. Think twice before generating a new example. New example (in JSON):"
    sample_num=5 if neg_sample else 1
    new_example=query_azure_openai_chatgpt_chat(template,temperature=0.7,n=sample_num)
    best_example=random.choice(new_examples) #best_example_selection(reward_prompt,new_examples, temperature=0)
    #pdb.set_trace()
    #for _ in range(sample_num):
    #    new_example=query_azure_openai_chatgpt_chat(template,temperature=0.7)
 #   pdb.set_trace()
     
    #if not example_check(new_example,obj_passage):
     #   return None,None
     #   new_examples.append(new_example)
    for re_instruction in revise_instruction:
        prompt=META_REVISION_PROMPT
        prompt+=re_instruction
        prompt+='The given_example:{0}'.format(new_example)
        if 'compos' in re_instruction:
            for id in range(len(high_quality_examples)):
                prompt+='Demo Example {0}: {1}'.format(id,high_quality_examples[id])
        prompt+='Directly output the new example.'
        new_example=query_azure_openai_chatgpt_chat(prompt,temperature=0.7)
        new_examples.append(new_example)
    
    new_examples_2=[example_check(example,obj_passage) for example in new_examples]
#    pdb.set_trace()
    new_examples=new_examples_2
    if len(new_examples)==0:
        return None,None
    return new_examples,example_check(best_example,obj_passage) #new_examples[-1]

def label_transform(label):
    if label==1:
        return 'neutral'
    if label==0:
        return 'entailment'
    if label==2:
        return 'contradiction'
def cot_check_fill(demo_examples):
    first_example=demo_examples[0]
    demo_cot_examples=[]
    prompt='Does the example has the specific solution (step by step) to get its final output? Directly output yes or no.'
    prompt+='Example:{0}'.format(first_example)
    cot_status=query_azure_openai_chatgpt_chat(prompt)
#    pdb.set_trace()
    if 'yes' in cot_status.lower():
        return demo_examples
    else:
        for example in demo_examples:
            prompt="Your task is to generate the specific solution (step by step) to get its final output. The solution starts with 'Let's think step by step' and ends with 'The final answer is ...'. Example:{0}. Directly and only output the text solution without the input.".format('Input: '+example['Input']+'Output: '+example['Output'])
            cot_solution=query_azure_openai_chatgpt_chat(prompt)
 #           pdb.set_trace()
            demo_cot_examples.append({'Input':example['Input'],'Output':cot_solution})

        return demo_cot_examples
def data_sample_pattern(instruction,domain,num,store_name,reward_prompt,temperature=0.7,task_name='nli',neg_sample=True,pattern=False,iteration_number=1,sample_demo_num=3,passage_num=5000,valid_num=100,types=None):
    task_instruction=instruction
    try:
        demo_cot_examples=torch.load('{0}_demo.pt'.format(task_name))
    except:
        demo_cot_examples=cot_check_fill(demo_examples)
        torch.save(demo_cot_examples,'{0}_demo.pt'.format(task_name))
    try:
        passages=torch.load('{0}.pt'.format(domain)) #'medical_passages_sub.pt')
    except:
        passages=prompt_domain_find_wikipassage(domain,passage_num)
        torch.save(passages,'{0}.pt'.format(domain)) #'medical_passages_sub.pt')
    passages=list(set(passages))
    #try:
    #    synthetic_examples=torch.load('{0}.pt'.format(store_name))
    #except:
    #    synthetic_examples=[]
    #start=len(synthetic_examples)
    prompts=[]
    for num_id in range(0,num+3):
        
        sample_size = 1 # random.randint(1,5)
        sampled_objects = passages[num_id]
        sampled_demo_examples=random.sample(demo_cot_examples, sample_demo_num)
        prompts.append(aug_few_shot_prompt_pattern_generate(sampled_demo_examples,task_instruction,sampled_objects,reward_prompt,neg_sample=neg_sample))   
    results=batch_inference(prompts,temperature=0.7)
    try:
        return [example_check(result) for result in results][:num] #examples=[example_check(result) for result in results][:num]
    except:
        results=batch_inference(prompts,temperature=0.7)
        return [example_check(result) for result in results][:num] #examples=[example_check(result) for result in results][:num]
    short_examples=[]
    random_examples=[]
    long_examples=[]
    for example_id,example in enumerate(examples):
        if not example:
            continue  #        pdb.set_trace()
        if example_id%100==0:
            print('We have done {0} examples.'.format(example_id))
        input=example['input']
        short_output,random_output,long_output=majority_voting(example)
        short_examples.append({'input':input,'output':short_output})
        random_examples.append({'input':input,'output':random_output})
        long_examples.append({'input':input,'output':long_output})
        #example['output'=majority_voting(example)
#        pdb.set_trace()
    return short_examples,random_examples,long_examples
def majority_voting(example,n=4):
    input=example['input']
    output=example['output']
    prompt="You should give an output to the query and use 'The final answer is xxx' to end your output."+input+"Let's think step by step."
    responses=query_azure_openai_chatgpt_chat(prompt,temperature=0.7,n=n)
    responses.append(output) 
    answers=[extract_answer_prediction_nli(response) for response in responses]
    mode_value = mode(answers)
    mode_ids = [index for index, value in enumerate(answers) if value == mode_value]
    selected_response=[(responses[index],len(responses[index])) for index in mode_ids]
    selected_response.sort(key=lambda x:x[1])
    
 #   pdb.set_trace()
    return selected_response[0][0],responses[random.choice(mode_ids)],selected_response[-1][0]
 
def data_sample(instruction,domain,num,store_name,reward_prompt,task_name='nli',neg_sample=True,pattern=False,iteration_number=1,sample_num=3,passage_num=5000,valid_num=100,types=None):

    # demo examples collection
    demo_examples=[]
    task_instruction=instruction
    #multi_nli=load_dataset('nyu-mll/multi_nli')
    dataset=load_dataset('hippocrates/MedNLI_test')  
    #for id in range(len(multi_nli['train'])): examples.append('INPUT: '+'Premise: '+multi_nli['train'][id]['premise']+'Hypothesis: '+multi_nli['train'][id]['hypothesis']+'Output'+label_transform(multi_nli['train'][id]['label']))
#    instruction='The domain is Medical. The TASK: Please classify the relationship between the given premise and hypothesis into one of the following labels: entailment, contradiction, or neutral. Return only the label.'
    for id in range(len(dataset['train'])):demo_examples.append({'Input':dataset['train'][id]['query'],'Output':dataset['train'][id]['answer']})
    demo_examples=demo_examples[:valid_num]
    #pdb.set_trace()
    try:
        demo_cot_examples=torch.load('{0}_demo.pt'.format(task_name))
    except:
        demo_cot_examples=cot_check_fill(demo_examples)
        torch.save(demo_cot_examples,'{0}_demo.pt'.format(task_name))
#    pdb.set_trace() #from collect_data import prompt_domain_find_wikipassage
    #passages=list(set(passages))
#    pdb.set_trace()
    try:
        passages=torch.load('{0}.pt'.format(domain)) #'medical_passages_sub.pt')
    except:
        passages=prompt_domain_find_wikipassage(domain,passage_num)
        torch.save(passages,'{0}.pt'.format(domain)) #'medical_passages_sub.pt')
    passages=list(set(passages)) #[3000:] #pdb.set_trace()
#    pdb.set_trace()
    try:
        synthetic_examples=torch.load('{0}.pt'.format(store_name))
    except:
        synthetic_examples=[]
    try:
        synthetic_examples_score=torch.load('{0}_score.pt'.format(store_name))
    except:
        synthetic_examples_score=[]
    try:
        synthetic_examples_all=torch.load('{0}_all.pt'.format(store_name))
    except:
        synthetic_examples_all=[]
    #pdb.set_trace()
    #types=["Complex Join Operations**: The model struggles with queries that require multiple table joins to retrieve the correct data, especially when the joins involve foreign key relationships and specific conditions across different tables.","**Nested Queries and Subqueries**: The model has difficulty generating correct SQL for nested queries and subqueries, particularly when the nested query involves conditions that need to be matched with the outer query.","**Specific Column Selection and Conditions**: The model often fails to select the correct columns or apply the correct conditions to columns,leading to incomplete or incorrect SQL statements. This includes issues with selecting columns that do not exist or misapplying conditions.","**Handling of Missing or Implicit Information**: The model struggles when the query requires understanding or inferring missing or implicit information, such as default values or implicit relationships between tables.","**Aggregation Functions and Grouping**: The model has issues correctly applying aggregation functions (e.g., SUM, MAX, COUNT) in the context of complex queries, especially when these functions need to be combined with other SQL operations such as grouping and ordering.","**Date and Time Handling**: The model has difficulty correctly interpreting and using date and time formats, which can lead to errors in queries that involve date comparisons or date-based conditions."] #    instruction=query_azure_openai_chatgpt_chat(generation_prompt)
#    pdb.set_trace() #generation_instruction=eval(generation_instruction) #pdb.set_trace()#basic necessary conditions:'{1}', and exceptions and errors:'{3}'
#    instruction=domain_knowledge+'\n'+instruction
#    instruction+='\nYou can refer to these data generation instructions:{0}'.format(generation_instruction)
#    pdb.set_trace()
    start=len(synthetic_examples)
    for num_id in range(start,num):
        
        sample_size = 1 # random.randint(1,5)
        sampled_objects = passages[num_id] #random.sample(passages, sample_size)
#        pdb.set_trace()
        errors=0
        
        operation_instructions = [
    "You can follow the structure of the demonstration examples but modify the content based on the related object or passage to create new training data.",
    "You can follow the problem type in demonstration examples but adjust the content according to the related object or passage to generate new training data.",
    "You can follow the reasoning type of the solution in demonstration examples but revise the content based on the related object or passage to build new training data.",
    "You can paraphrase the input and output of a demonstration example using the content of the related object or passage to form new training data.",
    "You can simplify the complexity of a demonstration example to construct new training data based on the related object or passage.",
    "You can increase the complexity of a demonstration example to create new training data based on the related object or passage.",
    "You can change the style, tone, or formality of the language in demonstration examples to generate new training data based on the related object or passage.",
    "You can introduce new elements or operations intentionally into a demonstration example to build new training data based on the related object or passage.",
    "You can explicitly request multi-step reasoning for demonstration examples to construct new training data based on the related object or passage.",
    "You can add more contextual details, constraints, or conditions to demonstration examples to form new training data based on the related object or passage.",
    "You can increase the depth and breadth of inquiry in demonstration examples to construct new training data based on the related object or passage.",
    "You can compose inquirys and solutions within demonstration examples to create new training data based on the related object or passage."
]
        revise_instruction=[
"the depth and breadth of the given example can be increased.",
"If the given example can be solved with just a few simple reasoning processes, you can rewrite it to explicitly request multiple-step reasoning.",
"Please add one more constraints/requirements into the given example",
"You can create a more complex example for the given example by referring to the demo examples.",
"This new example should belong to the same domain as the given example but be even more rare. The LENGTH and difficulty level of the new example should be similar to that of the given example."]
            
        op_instruction=random.sample(operation_instructions,1)
        re_instruction=[] #random.choices(revise_instruction,k=iteration_number)
        r_examples=random.sample(demo_cot_examples, sample_num)
        if types:
            choiced_type=random.sample(types,1)
            data_instruction=instruction+'You should focus on generating data with thess characteristic/type:{0}'.format(choiced_type)
        else:
            data_instruction=instruction
        print(data_instruction,'\n') #,op_instruction,'\n')
        if pattern:
            all_example,new_example=aug_few_shot_prompt_pattern_generate(r_examples,task_instruction,sampled_objects,reward_prompt,neg_sample=neg_sample,operation_instruction=op_instruction,revise_instruction=revise_instruction)
        else:
            all_example,new_example=aug_few_shot_prompt_generate(r_examples,task_instruction,sampled_objects,reward_prompt,neg_sample=neg_sample,operation_instruction=op_instruction,revise_instruction=re_instruction)
        
#                pdb.set_trace()
        if new_example: #pdb.set_trace()
            ''' #examples_ori_all.extend(all_example)
            evaluation_criteria='Your task is to predict the quality score of the given example based on its quality, instruction following, and tone. The score ranges from 0 to 5. Higher score means better quality. Directly output the score.'
            score=evaluate_score_reason(new_example,evaluation_criteria) 
            match = re.findall(r'\b[0-5]\b', score)
            number=int(match[0])if match else None
            synthetic_examples_score.append((new_example,number))
            '''
            synthetic_examples.append(new_example)
            #torch.save(examples_ori_all,'{0}_all.pt'.format(store_name))
            torch.save(synthetic_examples,'{0}.pt'.format(store_name))
            #torch.save(synthetic_examples_score,'{0}_score.pt'.format(store_name))
        if all_example:
            synthetic_examples_all.append(all_example)
            torch.save(synthetic_examples_all,'{0}_all.pt'.format(store_name))
            

        if synthetic_examples:
            print(synthetic_examples[-1])
        print(num_id,len(synthetic_examples_all),len(synthetic_examples),errors)
    try:
        #synthetic_examples_score=sorted(synthetic_examples_score,key=lambda x:x[1],reverse=True)
        #synthetic_examples_score=[data[0] for data in synthetic_examples_score]        
        clean_and_collect_dataset(synthetic_examples,'',store_name)
        #clean_and_collect_dataset(synthetic_examples_all,'',store_name+'_all')
        #clean_and_collect_dataset(synthetic_examples_score,'',store_name+'_score')
        return synthetic_examples,synthetic_examples_all #,synthetic_examples_score
    except:
        return synthetic_examples,synthetic_examples_all #,synthetic_examples_score
    
def sort_select_data(generated_examples):
    examples_num={'neutral':0,'entailment':0,'contradiction':0}
    generated_examples=sorted(generated_examples,key=lambda x:x[1],reverse=True)
    sorted_examples=[]
    for example in generated_examples:
        if examples_num[example[0]['output'].lower().replace('output','').strip()]>(num//len(['neutral', 'entailment', 'contradiction'])):
            continue
        else:
            sorted_examples.append(example[0])
            examples_num[example[0]['output'].lower().replace('output','').strip()]+=1
    return sorted_examples
def clean_input(input):
    if 'output' in input:
        input=input.split('output')[0]
    elif 'Output' in input:
        input=input.split('Output')[0]
    return input.strip()
def clean_output(output):
    if 'output' in output:
       output=output.replace('output','').replace(':','') #.split('output')[1]
    elif 'Output' in output:
        output=output.replace('Output','').replace(':','') #output.split('Output')[1]
    return output.strip()
def fixed_option_data_collect(examples,quota):
    options={}
    balanced_examples=[]
    for example in examples:
        if example['output'] not in options.keys():
            options[example['output']]=[]
            options[example['output']].append(example)
        else:
            options[example['output']].append(example)
    #pdb.set_trace()
    keys=list(options.keys())
    sorted_keys = sorted(keys, key=lambda k: len(options[k]))
    #pdb.set_trace()
    for key_id,key in enumerate(sorted_keys):
        if len(options[key])>=(quota//(len(sorted_keys)-key_id)):
            #quota-=(quota//(len(sorted_keys)-key_id))
            random.shuffle(options[key])
            balanced_examples.extend(options[key][:(quota//(len(sorted_keys)-key_id))])
            quota-=(quota//(len(sorted_keys)-key_id))
        else:
            quota-=len(options[key])
            balanced_examples.extend(options[key])
        #pdb.set_trace()
    return balanced_examples
def clean_and_collect_dataset(examples,instruction,name):
    if isinstance(examples[0],str):
        datas=[eval(ele[ele.find('{'):ele.find('}')+1]) for ele in examples]
    else:
        datas=examples
    train_data=[{'instruction':instruction+' '+clean_input(data['input']),'output':clean_output(data['output'])} for data in datas]
    dataset=Dataset.from_dict({key: [str(dic[key]) for dic in train_data] for key in train_data[0]})
#    dataset_dict = DatasetDict({'train': dataset})
    dataset=dataset.shuffle(seed=2022)
    dataset.save_to_disk(name)

def keys_above_average(dictionary):
    if not dictionary:
        return []
    
    values = list(dictionary.values())
    average = sum(values) / len(values)
    above_average_keys = [key for key, value in dictionary.items() if value > average]
    
    return above_average_keys
def multiple_examples_extraction(examples):
    generated_examples=[]
    for example in examples:
        clean_example=extract_example(str(example))
        if len(clean_example)>0:
            generated_examples.extend(clean_example)
    generated_examples = [elem for elem in generated_examples if elem]
    generated_examples = [elem[0] if isinstance(elem, list) else elem for elem in generated_examples]
    #generated_examples = [elem[0] for elem in generated_examples if isinstance(elem,list) else elem]
    return generated_examples
#    return examples
def extract_example(text):
    examples = []
    try:
        input_output = str(text).split('"input":')[1]
        if 'output' in input_output:
            input, output = input_output.split('output')[0], input_output.split('output')[1]
        elif 'Output' in input_output:
            input, output = input_output.split('Output')[0], input_output.split('Output')[1]
        else:
            return examples  # return empty list if no 'output' or 'Output' found
        
        new_example = {
            'input': input.replace(':', '').replace('{', '').replace('"', '').replace('\n', '').strip(),
            'output': output.replace(':', '').replace('"', '').replace('}', '').replace('\n', '').strip()
        }
        examples.append(new_example)
    except IndexError:
        return examples  # return empty list if 'input' not found
    
    return examples
def extract_examples(text):
    splits=text.split('}')
    raw_examples=[]
    for split in splits: 
        if 'input' in split or 'Input' in split:
            raw_examples.append(split)
    examples=[]
    for raw_example in raw_examples:
        input_tag='"Input":' if '"Input":' in str(raw_example) else "'Input':" #try:
        #    print(raw_example) #pdb.set_trace()
        input_output=str(raw_example).split(input_tag)[1]
 #       except:
#            pdb.set_trace()
        if 'output' in input_output:
            output_tag='"output":' if '"output":' in str(input_output) else "'output':"
            input,output=input_output.split(output_tag)[0],input_output.split(output_tag)[1]
        elif 'Output' in input_output:
            output_tag='"Output":' if '"Output":' in str(input_output) else "'Output':"
            input,output=input_output.split(output_tag)[0],input_output.split(output_tag)[1]
        else:
            continue
            pdb.set_trace()
        new_example={'input':input.replace(':','').replace('{','').replace('"','').replace('\n','').strip(),'output':output.replace(':','').replace('"','').replace('}','').replace('\\n','').replace('\n','').strip()}
        examples.append(new_example)
    return examples
