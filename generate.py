import pdb
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
def simple_few_shot_prompt_generate(high_quality_examples,instruction,too_many_keys,previous_examples=None):
    template=META_PROMPT
    template+="""Here is the requirement for the generation of new examples:{0}\n""".format(instruction)
    template+="Here are some [high-quality examples] for the task. You should follow their input and output format requirements. You should pay extreme attention to them!!!\n"
    HQ_examples = high_quality_examples #random.sample(high_quality_examples, 4)
    for example_id,example in enumerate(HQ_examples):
        template+="Example {0}: {1}\n".format(example_id+1,example)
    if previous_examples is not None:
        template+="These are some previous generated examples. You should avoid generate examples that are the same to them.\n"
        prev_examples = random.sample(previous_examples, min(4,len(previous_examples)))
        for example_id,example in enumerate(prev_examples):
            template+="Example {0}: {1}\n".format(example_id+1,example)
    if len(too_many_keys)>0:
        template+='Recently,we have too many samples with these outputs:{0}. Please avoid to generate samples with these outputs.'.format(too_many_keys)
    template+="Before generating one new examples, ensure that you strictly adhere to the rules mentioned in the [Requirement] and follow the input and output format of the [high-quality examples]. Think twice before generating a new example. [new example (in JSON)]:"
    return query_azure_openai_chatgpt_chat(template)
def example_check(new_example,obj_passage):
        new_example=new_example[new_example.find('{'):new_example.rfind('}')+1]
        try:
            new_example=extract_examples(new_example)[0]
        except:
            return
        #pdb.set_trace()
        try:
            new_example['input']="[INST] "+"""Here is a database schema:{0} """.format(obj_passage[0][1])+'None'+new_example['input']
        except:
            return
        return new_example
def aug_few_shot_prompt_generate(high_quality_examples,instruction,task_instruction,too_many_keys,obj_passage,pattern,domain_knowledge,operation_instruction=None,revise_instruction=None,revision=None,previous_examples=None):
    if revision is not None:
        template="Your task is to generate a higher quality example by revising the previous generated example based on its issue analysis and revision advice. Prev example and analysis:{0}".format(revision)
        template+="Here is a related passage that you can refer to. " #For example, you can rephrase it or utilize its domain knowledge in text to construct the new training data." # You can follow this procedu$
        template+="Passage:{0}".format(passage[:min(2048,len(passage))])
        return query_azure_openai_chatgpt_chat(template)
    else:
        template=META_PROMPT
    template+='You must consider the task instruction (task knowledge), provided examples (format), and the passage (domain knowledge) to generate your training data.'
    template+=""" Here is the task instruction:{0}\n""".format(instruction)
##    if revision is None:
  #      elements = ['neutral', 'entailment', 'contradiction']
   #     sampled_element = random.choice(elements)
#    pdb.set_trace()
    #    template+="""The output of your generated examples must be {0}.\n""".format(sampled_element)
    #prompt="""The task is '{0}', you can induce the its input and output format pattern in general from these examples:{1}. You only need to output the format pattern:""".format(instruction,high_quality_examples)
#        print(sampled_element)    #pattern=query_azure_openai_chatgpt_chat(prompt)
    template+=" Here is some demonstration examples. You should follow the examples strictly to generate data!!!" #and the input and output's format pattern in general:{1}. The output includes its solution steps and the final result. You should follow the pattern strictly to generate data!!!".format(high_quality_examples,pattern)
    #for id in range(len(high_quality_examples)):
    #    template+='Example {0}: {1}'.format(id,high_quality_examples[id])
    task_instruction+=" Here is some examples:{0}".format(high_quality_examples)
 #   elements = ['neutral', 'entailment', 'contradiction']
#    sampled_element = random.choice(elements)
#    pdb.set_trace() #template+="The knowledge about the task includes:{0}. You can refer it and your generated example's content should in this range".format(domain_knowledge)
    template+="Here is some related objects or passages that you can refer to."+operation_instruction[0]  #"For example, you can utilize its information to construct the new training data." # You can follow this procedure to construct the training sample:{0}.".format(procedure) # For example, you can rephrase it or utilize its domain knowledge in text to construct the new training data."
    for id in range(len(high_quality_examples)):
        template+='Demo Example {0}: {1}'.format(id,high_quality_examples[id])
    template+="Related Objects or Passages:{0}".format(obj_passage[0][1][:min(2048,len(obj_passage[0][1]))])
#    api_docs=""
    print(operation_instruction[0],'\n')
#    candidate_api_functions = api_retriever(' '.join(obj_passage))
#    api_docs += "I have the following set of API:\n\n"
#    for d in candidate_api_functions:
#        api_docs += d + "\n"
    new_examples=[]
    template+="Before generating the new example, ensure that you strictly adhere to the rules mentioned in the [Requirement] and follow the format of the [high-quality examples]. Think twice before generating a new example. New example (in JSON):"
    new_example=query_azure_openai_chatgpt_chat(template,temperature=0.7)
    if not example_check(new_example,obj_passage):
        return None,None
    new_examples.append(new_example)
    for re_instruction in revise_instruction:
        prompt=META_REVISION_PROMPT
        prompt+=re_instruction
        prompt+='The given_example:{0}'.format(new_example)
        if 'compos' in re_instruction:
            for id in range(len(high_quality_examples)):
                prompt+='Demo Example {0}: {1}'.format(id,high_quality_examples[id])
        prompt+='Directly output the new example.'
        new_example=query_azure_openai_chatgpt_chat(prompt,temperature=0.7)
#        pdb.set_trace()
        new_examples.append(new_example)
    
#    pdb.set_trace()
    #pdb.set_trace()    
#new_examples_2=[example_check(example,obj_passage) for example in new_examples] 
    new_examples_2=[example_check(example,obj_passage) for example in new_examples]
    #pdb.set_trace()
    new_examples=new_examples_2
    if len(new_examples)==0:
        return None,None
#new_example=new_example[new_example.find('{'):new_example.rfind('}')+1]
    ''' 
   try:
        instruction=eval(new_example)['input']
    except:
        instruction=new_example.split('input')[1].split('output')[0].split(':')[1].replace(',','').replace('\n','')
    task_instruction+=' There are the list of avaliable apis:{0}. You should use them to generate the output'.format(api_docs)
    task_instruction+=' Current user instruction:{0}. You only need to generate its output:'.format(instruction)
#    pdb.set_trace()
    output=query_azure_openai_chatgpt_chat(task_instruction)
    if 'Action:' in output:
         output=output.split('Action:')[1]
    '''
    #pdb.set_trace()
    import sqlite3
    db_id=obj_passage[0][0]
    path='/dccstor/obsidian_llm/yiduo/AgentBench/DAMO-ConvAI/bird/data/train/train_databases/{0}/{0}.sqlite'.format(db_id)
#    try:
 #       new_example=extract_examples(new_example)[0]
  #  except:
   #     return
 #   pdb.set_trace()
    #try:
    #    new_example['input']="[INST] "+"""Here is a database schema:{0} """.format(obj_passage[0][1])+'None'+new_example['input'].split("None")[1]
    #except:
    #    return
    #pdb.set_trace()
    def handler(signum, frame):
        raise TimeoutError("Query execution exceeded the time limit")
    def execute_verification(new_example,path):
        new_example['output']=new_example['output'].replace('sql','').replace('```','')
        #pdb.set_trace() # Set the alarm signal
        try:
            conn = sqlite3.connect(path)
            cursor = conn.cursor()
            sql='SELECT'+new_example['output'].split('SELECT')[-1].split(';')[0]
        #    sql=sql.replace('sql',''),replace('```','')
            cursor.execute(sql)
            result = cursor.fetchall()
            if not result:
                return
            conn.close()
            return new_example
        #   pdb.set_trace()
        except Exception as e:
            #a=1 #continue
            
            sql='SELECT'+new_example['output'].split('SELECT')[-1].split(';')[0]
            old_sql=sql
            prompt="""The current sql is {sql}, which I execute it, it return the error {e}.Your task is to fix the error by editing the sql and directly only return the edited sql.
        """.format(sql=sql,e=e)
            sql=query_azure_openai_chatgpt_chat(prompt).replace('sql','').replace('```','')
            conn = sqlite3.connect(path)
            cursor = conn.cursor()
#            pdb.set_trace()
            
            try:
                cursor.execute(sql)
                result = cursor.fetchall()
                if not result:
                    return
                conn.close()
            except:
                return
            new_example['output']=sql
            
            return #new_example
    results=[]
    #pdb.set_trace()
    for new_example in new_examples:
        try:    
            res =func_timeout(30,execute_verification,args=(new_example,path))        #pdb.set_trace()
            if res is not None:
                results.append(res)
        except FunctionTimedOut: 
            a=1 #return #pdb.set_trace()
#    pdb.set_trace()
    if len(results)==0:
        return None,None    
    return results,results[-1] #new_example #{"input":instruction,"output":'Action:'+output}) #new_example #query_azure_openai_chatgpt_chat(template) #,0.3)
#bio_nli=load_dataset('nyu-mll/multi_nli')
#from datasets import load_dataset
def label_transform(label):
    if label==1:
        return 'neutral'
    if label==0:
        return 'entailment'
    if label==2:
        return 'contradiction'
def inference_data_simple(instruction,num):
    examples=[]
    multi_nli=load_dataset('nyu-mll/multi_nli')
    dataset=load_dataset('hippocrates/MedNLI_test')  
    #for id in range(len(multi_nli['train'])): examples.append('INPUT: '+'Premise: '+multi_nli['train'][id]['premise']+'Hypothesis: '+multi_nli['train'][id]['hypothesis']+'Output'+label_transform(multi_n$
#    instruction='The domain is Medical. The TASK: Please classify the relationship between the given premise and hypothesis into one of the following labels: entailment, contradiction, or neutral. Retur$
    for id in range(len(dataset['train'])):examples.append('INPUT:'+dataset['train'][id]['query']+' Output:'+dataset['train'][id]['answer'])
    generated_examples=[]
    #from collect_data import prompt_domain_find_wikipassage
    #passages=prompt_domain_find_wikipassage('medical',2000)
    output_count={}
    examples=random.sample(examples,4)
    too_many_keys=[]
    for num_id in range(num):
        if generated_examples:
            new_example=simple_few_shot_prompt_generate(examples,instruction,too_many_keys, previous_examples=generated_examples)
            #generated_examples.append(new_example)
        else:
            new_example=simple_few_shot_prompt_generate(examples,instruction,too_many_keys)
            #generated_examples.append(new_example)
        try:
            new_example=eval(new_example[new_example.find('{'):new_example.find('}')+1])
            output=new_example['output']
        except:
            input_output=str(new_example).split('input')[1]
            if 'output' in input_output:
                input,output=input_output.split('output')[0],input_output.split('output')[1]
            else:
                input,output=input_output.split('Output')[0],input_output.split('Output')[1]
            new_example={'input':input.replace(':','').replace('"','').replace('\n','').strip(),'output':output.replace(':','').replace('"','').replace('}','').replace('\n','').strip()} 
        if new_example['output'] in output_count.keys():
            output_count[new_example['output']]+=1
        else:
            output_count[new_example['output']]=1
        too_many_keys=keys_above_average(output_count)
        
        generated_examples.append(new_example)
        print(num_id,new_example)
    return generated_examples
def inference_data(instruction,domain,num,store_name,sample_num=3,demos=None,types=None):
    examples=[]
    task_instruction=instruction
    #multi_nli=load_dataset('nyu-mll/multi_nli')
    #dataset=load_dataset('hippocrates/MedNLI_test')  
    #for id in range(len(multi_nli['train'])): examples.append('INPUT: '+'Premise: '+multi_nli['train'][id]['premise']+'Hypothesis: '+multi_nli['train'][id]['hypothesis']+'Output'+label_transform(multi_nli['train'][id]['label']))
#    instruction='The domain is Medical. The TASK: Please classify the relationship between the given premise and hypothesis into one of the following labels: entailment, contradiction, or neutral. Return only the label.'
    #for id in range(len(dataset['train'])):examples.append('INPUT:'+dataset['train'][id]['query']+' Output:'+dataset['train'][id]['answer'])
    generated_examples=[]
    #from collect_data import prompt_domain_find_wikipassage
    #passages=list(set(passages))
    try:
        passages=torch.load('databases.pt') #'medical_passages_sub.pt')
    except:
        passages=prompt_domain_find_wikipassage('databases',2000)
        torch.save(passages,'hhh.pt') #'medical_passages_sub.pt')
    passages=list(set(passages)) #pdb.set_trace()
    #output_count={}
    #pdb.set_trace()
    #objects=set()
    #import json
    #with open('/dccstor/obsidian_llm/yiduo/AgentBench/toolbench/data/virtual_home/v0/valid_object_lists.json') as f:
    #    file=json.load(f) 
    #for api in file: 
    #    for obj_key in file[api]:
    #        objs=file[api][obj_key]
    #        for obj in objs:
    #            objects.add(obj)
    #objects=list(objects)
    #pdb.set_trace()
#pdb.set_trace()
    #demo_examples=['Task: Put on coat and shoes\nAction:\nAgent.Find(shoes)\nAgent.TurnTo(shoes)\nAgent.PointAt(shoes)\nAgent.Wash(shoes)\n','Task: Read book\nAction:\nAgent.Find(desk)\nAgent.Find(chair)\nAgent.Find(novel)\nAgent.Grab(novel)\nAgent.Read(novel)\n','Task: Read book\nAction:\nAgent.WalkTo(bedroom)\nAgent.WalkTo(bed)\nAgent.Find(bed)\nAgent.LieOn(bed)\nAgent.Find(novel)\nAgent.Grab(novel)\nAgent.Read(novel)\n','Task: Get ready to leave\nAction:\nAgent.WalkTo(home_office)\nAgent.WalkTo(shoes)\nAgent.Find(shoes)\nAgent.Grab(shoes)\nAgent.PutOn(shoes)\nAgent.Find(keys)\nAgent.Grab(keys)\n'] #"I have the following set of API:# Read an object. 'object' can only be: ['address_book', 'homework', 'mail', 'novel', 'check', 'newspaper']. \nAgent.Read(object)\n # Drink from an object. 'object' can only be: ['water_glass', 'cup', 'juice', 'coffee', 'drinking_glass', 'coffee_cup', 'beer']. \nAgent.Drink(object)\n # Wipe an object. 'object' can only be: ['face', 'vacuum_cleaner', 'mirror', 'chair', 'toilet_paper', 'filing_cabinet', 'desk', 'hair', 'sink', 'computer', 'television', 'faucet', 'floor_lamp', 'hands_both', 'drinking_glass', 'table', 'kitchen_counter', 'instrument_piano', 'coffee_table', 'woman', 'dirt', 'couch', 'plate', 'bowl', 'towel']. \nAgent.Wipe(object)\n # Walk to to a room or object. 'object' can only be: ['coffee_pot', 'cup', 'button', 'dresser', 'milk', 'phone', 'cutting_board', 'water', 'bed', 'video_game_console', 'soap', 'filing_cabinet', 'oven', 'shredder', 'address_book', 'faucet', 'coffee', 'newspaper', 'mop_bucket', 'basket_for_clothes', 'alarm_clock', 'shampoo', 'bowl', 'laptop', 'towel', 'board_game', 'man', 'food_bread', 'laundry_detergent', 'freezer', 'bookshelf', 'cupboard', 'chair', 'water_glass', 'floor', 'brush', 'desk', 'kitchen_cabinet', 'fly', 'toy', 'curtain', 'computer', 'television', 'microwave', 'mat', 'fax_machine', 'clothes_jacket', 'dirt', 'shower', 'plate', 'coffee_filter', 'clothes_pants', 'pajamas', 'after_shave', 'bills', 'child', 'keys', 'window', 'wall', 'table_cloth', 'trashcan', 'coffee_maker', 'nightstand', 'hanger', 'cat', 'food_food', 'floor_lamp', 'home_office', 'pillow', 'controller', 'electrical_outlet', 'dry_pasta', 'washing_machine', 'dining_room', 'toilet', 'stereo', 'coffee_table', 'paper_towel', 'clothes_dress', 'couch', 'purse', 'electric_shaver', 'drawing', 'rag', 'remote_control', 'mop', 'vacuum_cleaner', 'light', 'shoes', 'pot', 'mirror', 'dish_soap', 'cleaning_solution', 'bedroom', 'bathroom_cabinet', 'sink', 'novel', 'detergent', 'drinking_glass', 'bathroom', 'table', 'kitchen_counter', 'wall_clock', 'woman', 'dishwasher', 'dog', 'fork']. \nAgent.WalkTo(object)\n # Point at an object. 'object' can only be: ['cup', 'dresser', 'phone', 'shoes', 'bed', 'water', 'soap', 'folder', 'brush', 'oven', 'toy', 'mouse', 'cleaning_bottle', 'spoon', 'computer', 'television', 'cat', 'novel', 'faucet', 'razor', 'newspaper', 'table', 'couch', 'remote_control', 'plate', 'towel']. \nAgent.PointAt(object)\n # Squeeze an object. 'object' can only be: ['sponge', 'shampoo', 'soap', 'dish_soap', 'rag', 'cleaning_solution', 'towel']. \nAgent.Squeeze(object)\n # Close an object. 'object' can only be: ['cupboard', 'bookshelf', 'trashcan', 'coffee_maker', 'dresser', 'bathroom_cabinet', 'basket_for_clothes', 'mail', 'bag', 'folder', 'microwave', 'filing_cabinet', 'dishwasher', 'purse', 'freezer', 'washing_machine', 'kitchen_cabinet']. \nAgent.Close(object)\n # Turn an object on. 'object' can only be: ['vacuum_cleaner', 'console', 'light', 'phone', 'video_game_console', 'cd_player', 'oven', 'shredder', 'coffee_maker', 'television', 'computer', 'faucet', 'floor_lamp', 'toaster', 'electrical_outlet', 'washing_machine', 'stereo', 'microwave', 'fax_machine', 'dishwasher', 'electric_shaver', 'remote_control', 'laptop']. \nAgent.SwitchOn(object)\n # Stand up. \nAgent.StandUp()\n # Watch computer or television. 'object' can only be: ['television', 'computer']. \nAgent.Watch(object)\n Task: Read book\nAction:\nAgent.Find(desk)\nAgent.Find(chair)\nAgent.Find(novel)\nAgent.Grab(novel)\nAgent.Read(novel)","I have the following set of API:# Drink from an object. 'object' can only be: ['water_glass', 'cup', 'juice', 'coffee', 'drinking_glass', 'coffee_cup', 'beer']. \nAgent.Drink(object)\n # Wipe an object. 'object' can only be: ['face', 'vacuum_cleaner', 'mirror', 'chair', 'toilet_paper', 'filing_cabinet', 'desk', 'hair', 'sink', 'computer', 'television', 'faucet', 'floor_lamp', 'hands_both', 'drinking_glass', 'table', 'kitchen_counter', 'instrument_piano', 'coffee_table', 'woman', 'dirt', 'couch', 'plate', 'bowl', 'towel']. \nAgent.Wipe(object)\n # Walk to to a room or object. 'object' can only be: ['coffee_pot', 'cup', 'button', 'dresser', 'milk', 'phone', 'cutting_board', 'water', 'bed', 'video_game_console', 'soap', 'filing_cabinet', 'oven', 'shredder', 'address_book', 'faucet', 'coffee', 'newspaper', 'mop_bucket', 'basket_for_clothes', 'alarm_clock', 'shampoo', 'bowl', 'laptop', 'towel', 'board_game', 'man', 'food_bread', 'laundry_detergent', 'freezer', 'bookshelf', 'cupboard', 'chair', 'water_glass', 'floor', 'brush', 'desk', 'kitchen_cabinet', 'fly', 'toy', 'curtain', 'computer', 'television', 'microwave', 'mat', 'fax_machine', 'clothes_jacket', 'dirt', 'shower', 'plate', 'coffee_filter', 'clothes_pants', 'pajamas', 'after_shave', 'bills', 'child', 'keys', 'window', 'wall', 'table_cloth', 'trashcan', 'coffee_maker', 'nightstand', 'hanger', 'cat', 'food_food', 'floor_lamp', 'home_office', 'pillow', 'controller', 'electrical_outlet', 'dry_pasta', 'washing_machine', 'dining_room', 'toilet', 'stereo', 'coffee_table', 'paper_towel', 'clothes_dress', 'couch', 'purse', 'electric_shaver', 'drawing', 'rag', 'remote_control', 'mop', 'vacuum_cleaner', 'light', 'shoes', 'pot', 'mirror', 'dish_soap', 'cleaning_solution', 'bedroom', 'bathroom_cabinet', 'sink', 'novel', 'detergent', 'drinking_glass', 'bathroom', 'table', 'kitchen_counter', 'wall_clock', 'woman', 'dishwasher', 'dog', 'fork']. \nAgent.WalkTo(object)\n # Point at an object. 'object' can only be: ['cup', 'dresser', 'phone', 'shoes', 'bed', 'water', 'soap', 'folder', 'brush', 'oven', 'toy', 'mouse', 'cleaning_bottle', 'spoon', 'computer', 'television', 'cat', 'novel', 'faucet', 'razor', 'newspaper', 'table', 'couch', 'remote_control', 'plate', 'towel']. \nAgent.PointAt(object)\n # Squeeze an object. 'object' can only be: ['sponge', 'shampoo', 'soap', 'dish_soap', 'rag', 'cleaning_solution', 'towel']. \nAgent.Squeeze(object)\n # Close an object. 'object' can only be: ['cupboard', 'bookshelf', 'trashcan', 'coffee_maker', 'dresser', 'bathroom_cabinet', 'basket_for_clothes', 'mail', 'bag', 'folder', 'microwave', 'filing_cabinet', 'dishwasher', 'purse', 'freezer', 'washing_machine', 'kitchen_cabinet']. \nAgent.Close(object)\n # Read an object. 'object' can only be: ['address_book', 'homework', 'mail', 'novel', 'check', 'newspaper']. \nAgent.Read(object)\n # Turn an object on. 'object' can only be: ['vacuum_cleaner', 'console', 'light', 'phone', 'video_game_console', 'cd_player', 'oven', 'shredder', 'coffee_maker', 'television', 'computer', 'faucet', 'floor_lamp', 'toaster', 'electrical_outlet', 'washing_machine', 'stereo', 'microwave', 'fax_machine', 'dishwasher', 'electric_shaver', 'remote_control', 'laptop']. \nAgent.SwitchOn(object)\n # Stand up. \nAgent.StandUp()\n # Watch computer or television. 'object' can only be: ['television', 'computer']. \nAgent.Watch(object)\n Task: Change TV channel\nAction:\nAgent.WalkTo(home_office)\nAgent.WalkTo(remote_control)\nAgent.Find(remote_control)\nAgent.Grab(remote_control)\nAgent.Find(button)\nAgent.Push(button)\nAgent.Find(television)\nAgent.TurnTo(television)\nAgent.LookAt(television)","I have the following set of API:# Run to a room or object. 'object' can only be: ['table', 'laptop', 'water', 'chair', 'cup', 'dining_room', 'toilet', 'sink', 'home_office', 'washing_machine', 'bathroom', 'bedroom']. \nAgent.RunTo(object)\n # Walk to to a room or object. 'object' can only be: ['coffee_pot', 'cup', 'button', 'dresser', 'milk', 'phone', 'cutting_board', 'water', 'bed', 'video_game_console', 'soap', 'filing_cabinet', 'oven', 'shredder', 'address_book', 'faucet', 'coffee', 'newspaper', 'mop_bucket', 'basket_for_clothes', 'alarm_clock', 'shampoo', 'bowl', 'laptop', 'towel', 'board_game', 'man', 'food_bread', 'laundry_detergent', 'freezer', 'bookshelf', 'cupboard', 'chair', 'water_glass', 'floor', 'brush', 'desk', 'kitchen_cabinet', 'fly', 'toy', 'curtain', 'computer', 'television', 'microwave', 'mat', 'fax_machine', 'clothes_jacket', 'dirt', 'shower', 'plate', 'coffee_filter', 'clothes_pants', 'pajamas', 'after_shave', 'bills', 'child', 'keys', 'window', 'wall', 'table_cloth', 'trashcan', 'coffee_maker', 'nightstand', 'hanger', 'cat', 'food_food', 'floor_lamp', 'home_office', 'pillow', 'controller', 'electrical_outlet', 'dry_pasta', 'washing_machine', 'dining_room', 'toilet', 'stereo', 'coffee_table', 'paper_towel', 'clothes_dress', 'couch', 'purse', 'electric_shaver', 'drawing', 'rag', 'remote_control', 'mop', 'vacuum_cleaner', 'light', 'shoes', 'pot', 'mirror', 'dish_soap', 'cleaning_solution', 'bedroom', 'bathroom_cabinet', 'sink', 'novel', 'detergent', 'drinking_glass', 'bathroom', 'table', 'kitchen_counter', 'wall_clock', 'woman', 'dishwasher', 'dog', 'fork']. \nAgent.WalkTo(object)\n # Turn to an object. 'object' can only be: ['cup', 'coffee_pot', 'button', 'dresser', 'milk', 'phone', 'water', 'bed', 'soap', 'mail', 'picture', 'filing_cabinet', 'oven', 'faucet', 'newspaper', 'food_cheese', 'towel', 'spoon', 'laptop', 'man', 'food_bread', 'dishrack', 'keyboard', 'freezer', 'bookshelf', 'chair', 'water_glass', 'brush', 'centerpiece', 'desk', 'kitchen_cabinet', 'fly', 'toy', 'mouse', 'homework', 'computer', 'television', 'shower', 'plate', 'clothes_pants', 'child', 'food_dessert', 'folder', 'window', 'hair', 'coffee_maker', 'hanger', 'cat', 'food_food', 'floor_lamp', 'creditcard', 'razor', 'electrical_outlet', 'washing_machine', 'toilet', 'coffee_table', 'paper_towel', 'clothes_dress', 'couch', 'drawing', 'remote_control', 'face', 'food_vegetable', 'vacuum_cleaner', 'light', 'shoes', 'pot', 'mirror', 'duster', 'cleaning_bottle', 'sink', 'toaster', 'novel', 'clothes_shirt', 'table', 'kitchen_counter', 'wall_clock', 'woman', 'dishwasher', 'dog', 'check']. \nAgent.TurnTo(object)\n # Sit on an object. 'object' can only be: ['bed', 'chair', 'toilet', 'couch', 'love_seat']. \nAgent.SitOn(object)\n # Scrub an object. 'object' can only be: ['mop', 'cup', 'toilet', 'plate', 'soap', 'sink', 'spoon', 'cat', 'shower', 'dishwasher', 'hands_both', 'drinking_glass', 'bowl', 'towel']. \nAgent.Scrub(object)\n # Open an object. 'object' can only be: ['coffee_pot', 'dresser', 'freezer', 'bookshelf', 'cupboard', 'folder', 'mail', 'filing_cabinet', 'window', 'oven', 'desk', 'kitchen_cabinet', 'trashcan', 'coffee_maker', 'curtain', 'bathroom_cabinet', 'nightstand', 'bag', 'washing_machine', 'toilet', 'basket_for_clothes', 'microwave', 'dishwasher', 'purse']. \nAgent.Open(object)\n # Rinse an object. 'object' can only be: ['cup', 'pot', 'water', 'water_glass', 'sponge', 'soap', 'towel', 'dish_soap', 'oven', 'cleaning_solution', 'knife', 'spoon', 'sink', 'faucet', 'clothes_underwear', 'detergent', 'drinking_glass', 'hands_both', 'toilet', 'shower', 'rag', 'plate', 'bowl', 'fork']. \nAgent.Rinse(object)\n # Wash an object. 'object' can only be: ['face', 'cup', 'food_vegetable', 'dresser', 'fork', 'shoes', 'child', 'coffee_cup', 'bed', 'water', 'soap', 'duster', 'brush', 'bathtub', 'toy', 'cleaning_bottle', 'hair', 'sink', 'razor', 'hands_both', 'drinking_glass', 'table', 'toilet', 'basket_for_clothes', 'shower', 'dishwasher', 'plate', 'bowl', 'spoon']. \nAgent.Wash(object)\n # Look at an object. 'object' can only be: ['face', 'clothes_pants', 'cup', 'man', 'food_vegetable', 'button', 'dresser', 'vacuum_cleaner', 'food_bread', 'milk', 'dishrack', 'light', 'phone', 'keyboard', 'freezer', 'child', 'pot', 'food_dessert', 'mirror', 'water', 'chair', 'water_glass', 'bed', 'soap', 'duster', 'towel', 'picture', 'window', 'filing_cabinet', 'brush', 'oven', 'centerpiece', 'desk', 'kitchen_cabinet', 'fly', 'toy', 'cleaning_bottle', 'hair', 'coffee_maker', 'check', 'homework', 'sink', 'hanger', 'computer', 'television', 'cat', 'novel', 'food_food', 'floor_lamp', 'spoon', 'faucet', 'toaster', 'electrical_outlet', 'creditcard', 'newspaper', 'razor', 'washing_machine', 'clothes_shirt', 'table', 'kitchen_counter', 'wall_clock', 'toilet', 'coffee_table', 'woman', 'paper_towel', 'clothes_dress', 'couch', 'food_cheese', 'dog', 'drawing', 'remote_control', 'plate', 'laptop']. \nAgent.LookAt(object)\n # Put object1 on object2. 'object1' can only be: ['coffee_pot', 'cup', 'sauce_pan', 'milk', 'coffee_cup', 'cutting_board', 'water', 'soap', 'picture', 'clothes_hat', 'cloth_napkin', 'coffee', 'bookmark', 'newspaper', 'mop_bucket', 'chef_knife', 'basket_for_clothes', 'food_cheese', 'bowl', 'laptop', 'spoon', 'board_game', 'food_bread', 'dishrack', 'laundry_detergent', 'keyboard', 'water_glass', 'printing_paper', 'chair', 'sponge', 'brush', 'toy', 'mouse', 'food_chicken', 'clothes_socks', 'clothes_jacket', 'plate', 'ground_coffee', 'coffee_filter', 'clothes_pants', 'clothes_skirt', 'after_shave', 'food_egg', 'food_oatmeal', 'napkin', 'keys', 'folder', 'toilet_paper', 'sheets', 'table_cloth', 'food_food', 'pillow', 'creditcard', 'dry_pasta', 'paper_towel', 'clothes_dress', 'remote_control', 'rag', 'coin', 'mop', 'food_vegetable', 'food_turkey', 'shoes', 'light_bulb', 'pot', 'dish_soap', 'cleaning_solution', 'pasta', 'knife', 'clothes_underwear', 'novel', 'detergent', 'drinking_glass', 'clothes_shirt', 'toothbrush', 'check', 'fork']. 'object2' can only be: ['mop', 'coffee_pot', 'cup', 'sauce_pan', 'dresser', 'dishrack', 'freezer', 'pot', 'bookshelf', 'cutting_board', 'cupboard', 'bed', 'chair', 'mousepad', 'floor', 'filing_cabinet', 'oven', 'desk', 'shredder', 'wall', 'trashcan', 'coffee_maker', 'nightstand', 'sink', 'hanger', 'toaster', 'garbage_can', 'novel', 'bag', 'washing_machine', 'table', 'kitchen_counter', 'mop_bucket', 'toilet', 'coffee_table', 'basket_for_clothes', 'fax_machine', 'dishwasher', 'couch', 'plate', 'drying_rack', 'bowl']. \nAgent.Put(object1, object2)\n Task: Go to toilet\nAction:\nAgent.WalkTo(bathroom)\nAgent.RunTo(toilet)","I have the following set of API:# Put a piece of clothing or accessary on. 'object' can only be: ['headset', 'spectacles', 'pajamas', 'clothes_socks', 'clothes_jacket', 'clothes_scarf', 'clothes_dress', 'shoes', 'clothes_shirt', 'clothes_pants']. \nAgent.PutOn(object)\n # Put object1 on object2. 'object1' can only be: ['coffee_pot', 'cup', 'sauce_pan', 'milk', 'coffee_cup', 'cutting_board', 'water', 'soap', 'picture', 'clothes_hat', 'cloth_napkin', 'coffee', 'bookmark', 'newspaper', 'mop_bucket', 'chef_knife', 'basket_for_clothes', 'food_cheese', 'bowl', 'laptop', 'spoon', 'board_game', 'food_bread', 'dishrack', 'laundry_detergent', 'keyboard', 'water_glass', 'printing_paper', 'chair', 'sponge', 'brush', 'toy', 'mouse', 'food_chicken', 'clothes_socks', 'clothes_jacket', 'plate', 'ground_coffee', 'coffee_filter', 'clothes_pants', 'clothes_skirt', 'after_shave', 'food_egg', 'food_oatmeal', 'napkin', 'keys', 'folder', 'toilet_paper', 'sheets', 'table_cloth', 'food_food', 'pillow', 'creditcard', 'dry_pasta', 'paper_towel', 'clothes_dress', 'remote_control', 'rag', 'coin', 'mop', 'food_vegetable', 'food_turkey', 'shoes', 'light_bulb', 'pot', 'dish_soap', 'cleaning_solution', 'pasta', 'knife', 'clothes_underwear', 'novel', 'detergent', 'drinking_glass', 'clothes_shirt', 'toothbrush', 'check', 'fork']. 'object2' can only be: ['mop', 'coffee_pot', 'cup', 'sauce_pan', 'dresser', 'dishrack', 'freezer', 'pot', 'bookshelf', 'cutting_board', 'cupboard', 'bed', 'chair', 'mousepad', 'floor', 'filing_cabinet', 'oven', 'desk', 'shredder', 'wall', 'trashcan', 'coffee_maker', 'nightstand', 'sink', 'hanger', 'toaster', 'garbage_can', 'novel', 'bag', 'washing_machine', 'table', 'kitchen_counter', 'mop_bucket', 'toilet', 'coffee_table', 'basket_for_clothes', 'fax_machine', 'dishwasher', 'couch', 'plate', 'drying_rack', 'bowl']. \nAgent.Put(object1, object2)\n # Put an object inside some other object. 'object1' can only be: ['food_vegetable', 'food_cereal', 'vacuum_cleaner', 'milk', 'food_egg', 'food_turkey', 'food_oatmeal', 'shoes', 'bills', 'food_fish', 'folder', 'dish_soap', 'food_pizza', 'knife', 'hanger', 'food_chicken', 'food_food', 'spoon', 'juice', 'food_carrot', 'clothes_shirt', 'clothes_jacket', 'clothes_dress', 'food_cheese', 'food_apple', 'check', 'plate', 'bowl', 'fork']. 'object2' can only be: ['dresser', 'folder', 'microwave', 'filing_cabinet', 'oven', 'freezer', 'kitchen_cabinet']. \nAgent.PutIn(object1, object2)\n # Type on an object. 'object' can only be: ['keyboard', 'phone']. \nAgent.TypeOn(object)\n # Lie on an object. 'object' can only be: ['bed', 'couch']. \nAgent.LieOn(object)\n # Sit on an object. 'object' can only be: ['bed', 'chair', 'toilet', 'couch', 'love_seat']. \nAgent.SitOn(object)\n # Put back an object. 'object' can only be: ['mop', 'cup', 'food_bread', 'milk', 'laundry_detergent', 'dishrack', 'food_oatmeal', 'phone', 'video_game_controller', 'pot', 'shampoo', 'sponge', 'water_glass', 'remote_control', 'bowl', 'soap', 'dish_soap', 'sheets', 'cleaning_solution', 'laptop', 'mouse', 'knife', 'juice', 'food_food', 'pillow', 'novel', 'coffee', 'drinking_glass', 'chef_knife', 'face_soap', 'basket_for_clothes', 'alarm_clock', 'broom', 'food_cheese', 'electric_shaver', 'rag', 'plate', 'ground_coffee', 'towel']. \nAgent.PutBack(object)\n # Turn an object on. 'object' can only be: ['vacuum_cleaner', 'console', 'light', 'phone', 'video_game_console', 'cd_player', 'oven', 'shredder', 'coffee_maker', 'television', 'computer', 'faucet', 'floor_lamp', 'toaster', 'electrical_outlet', 'washing_machine', 'stereo', 'microwave', 'fax_machine', 'dishwasher', 'electric_shaver', 'remote_control', 'laptop']. \nAgent.SwitchOn(object)\n # Take a piece of clothes off. 'object' can only be: ['clothes_jacket', 'clothes_dress', 'clothes_hat', 'shoes', 'clothes_shirt', 'clothes_pants']. \nAgent.TakeOff(object)\n # Pull an object. 'object' can only be: ['table', 'mop', 'mouse', 'chair', 'clothes_pants', 'light_bulb', 'curtain', 'vacuum_cleaner', 'mat', 'cat', 'food_food', 'drawing', 'shoes', 'centerpiece', 'sheets', 'pot', 'laptop']. \nAgent.Pull(object)\n Task: Put on coat and shoes\nAction:\nAgent.Find(shoes)\nAgent.TurnTo(shoes)\nAgent.PointAt(shoes)\nAgent.Wash(shoes)\n"] #random.sample(examples,4)
    demo_examples=["Input:[INST] Here is a database schema: lists : user_id [ INTEGER ] lists.user_id = lists_users.user_id list_id [ INTEGER ] primary_key list_title [ TEXT ] list_movie_number [ INTEGER ] list_update_timestamp_utc [ TEXT ] list_creation_timestamp_utc [ TEXT ] list_followers [ INTEGER ] list_url [ TEXT ] list_comments [ INTEGER ] list_description [ TEXT ] list_cover_image_url [ TEXT ] list_first_image_url [ TEXT ] list_second_image_url [ TEXT ] list_third_image_url [ TEXT ] movies : movie_id [ INTEGER ] primary_key movie_title [ TEXT ] movie_release_year [ INTEGER ] movie_url [ TEXT ] movie_title_language [ TEXT ] movie_popularity [ INTEGER ] movie_image_url [ TEXT ] director_id [ TEXT ] director_name [ TEXT ] director_url [ TEXT ] ratings_users : user_id [ INTEGER ] ratings_users.user_id = lists_users.user_id rating_date_utc [ TEXT ] user_trialist [ INTEGER ] user_subscriber [ INTEGER ] user_avatar_image_url [ TEXT ] user_cover_image_url [ TEXT ] user_eligible_for_trial [ INTEGER ] user_has_payment_method [ INTEGER ] lists_users : user_id [ INTEGER ] lists_users.user_id = lists.user_id list_id [ INTEGER ] lists_users.list_id = lists.list_id list_update_date_utc [ TEXT ] list_creation_date_utc [ TEXT ] user_trialist [ INTEGER ] user_subscriber [ INTEGER ] user_avatar_image_url [ TEXT ] user_cover_image_url [ TEXT ] user_eligible_for_trial [ TEXT ] user_has_payment_method [ TEXT ] ratings : movie_id [ INTEGER ] ratings.movie_id = movies.movie_id rating_id [ INTEGER ] rating_url [ TEXT ] rating_score [ INTEGER ] rating_timestamp_utc [ TEXT ] critic [ TEXT ] critic_likes [ INTEGER ] critic_comments [ INTEGER ] user_id [ INTEGER ] ratings.user_id = lists_users.user_id user_trialist [ INTEGER ] user_subscriber [ INTEGER ] user_eligible_for_trial [ INTEGER ] user_has_payment_method [ INTEGER ] External Knowledge: None\n-- Using valid SQLite and understading External Knowledge, answer the following questions for the tables provided above.\n-- Name movie titles released in year 1945. Sort the listing by the descending order of movie popularity?\nGenerate the SQL after thinking step by step: \nSELECT '  Output:SELECT movie_title FROM movies WHERE movie_release_year = 1945 ORDER BY movie_popularity DESC LIMIT 1;"]
    demo_examples.append("Input:[INST] Here is a database schema: authors : au_id [ TEXT ] primary_key au_lname [ TEXT ] au_fname [ TEXT ] phone [ TEXT ] address [ TEXT ] city [ TEXT ] state [ TEXT ] zip [ TEXT ] contract [ TEXT ] jobs : job_id [ INTEGER ] primary_key job_desc [ TEXT ] min_lvl [ INTEGER ] max_lvl [ INTEGER ] publishers : pub_id [ TEXT ] primary_key pub_name [ TEXT ] city [ TEXT ] state [ TEXT ] country [ TEXT ] employee : emp_id [ TEXT ] primary_key fname [ TEXT ] minit [ TEXT ] lname [ TEXT ] job_id [ INTEGER ] employee.job_id = jobs.job_id job_lvl [ INTEGER ] pub_id [ TEXT ] employee.pub_id = publishers.pub_id hire_date [ DATETIME ] pub_info : pub_id [ TEXT ] primary_key pub_info.pub_id = publishers.pub_id logo [ BLOB ] pr_info [ TEXT ] stores : stor_id [ TEXT ] primary_key stor_name [ TEXT ] stor_address [ TEXT ] city [ TEXT ] state [ TEXT ] zip [ TEXT ] discounts : discounttype [ TEXT ] stor_id [ TEXT ] discounts.stor_id = stores.stor_id lowqty [ INTEGER ] highqty [ INTEGER ] discount [ REAL ] titles : title_id [ TEXT ] primary_key title [ TEXT ] type [ TEXT ] pub_id [ TEXT ] titles.pub_id = publishers.pub_id price [ REAL ] advance [ REAL ] royalty [ INTEGER ] ytd_sales [ INTEGER ] notes [ TEXT ] pubdate [ DATETIME ] roysched : title_id [ TEXT ] roysched.title_id = titles.title_id lorange [ INTEGER ] hirange [ INTEGER ] royalty [ INTEGER ] sales : stor_id [ TEXT ] sales.stor_id = stores.stor_id ord_num [ TEXT ] ord_date [ DATETIME ] qty [ INTEGER ] payterms [ TEXT ] title_id [ TEXT ] sales.title_id = titles.title_id titleauthor : au_id [ TEXT ] titleauthor.au_id = authors.au_id title_id [ TEXT ] titleauthor.title_id = titles.title_id au_ord [ INTEGER ] royaltyper [ INTEGER ] External Knowledge: None\n-- Using valid SQLite and understading External Knowledge, answer the following questions for the tables provided above.\n-- Please list the first names of the employees who work as Managing Editor. \nGenerate the SQL after thinking step by step: \nSELECT '  Output: SELECT T1.fname FROM employee AS T1 INNER JOIN jobs AS T2 ON T1.job_id = T2.job_id WHERE T2.job_desc = 'Managing Editor';")
    demo_examples.append("Input:[INST] Here is a database schema: state : StateCode [ TEXT ] primary_key State [ TEXT ] Region [ TEXT ] callcenterlogs : Date received [ DATE ] Complaint ID [ TEXT ] primary_key rand client [ TEXT ] callcenterlogs.rand client = client.client_id phonefinal [ TEXT ] vru+line [ TEXT ] call_id [ INTEGER ] priority [ INTEGER ] type [ TEXT ] outcome [ TEXT ] server [ TEXT ] ser_start [ TEXT ] ser_exit [ TEXT ] ser_time [ TEXT ] client : client_id [ TEXT ] primary_key sex [ TEXT ] day [ INTEGER ] month [ INTEGER ] year [ INTEGER ] age [ INTEGER ] social [ TEXT ] first [ TEXT ] middle [ TEXT ] last [ TEXT ] phone [ TEXT ] email [ TEXT ] address_1 [ TEXT ] address_2 [ TEXT ] city [ TEXT ] state [ TEXT ] zipcode [ INTEGER ] district_id [ INTEGER ] client.district_id = district.district_id district : district_id [ INTEGER ] primary_key city [ TEXT ] state_abbrev [ TEXT ] district.state_abbrev = state.StateCode division [ TEXT ] events : Date received [ DATE ] Product [ TEXT ] Sub-product [ TEXT ] Issue [ TEXT ] Sub-issue [ TEXT ] Consumer complaint narrative [ TEXT ] Tags [ TEXT ] Consumer consent provided? [ TEXT ] Submitted via [ TEXT ] Date sent to company [ TEXT ] Company response to consumer [ TEXT ] Timely response? [ TEXT ] Consumer disputed? [ TEXT ] Complaint ID [ TEXT ] events.Complaint ID = callcenterlogs.Complaint ID Client_ID [ TEXT ] events.Client_ID = client.client_id reviews : Date [ DATE ] primary_key Stars [ INTEGER ] Reviews [ TEXT ] Product [ TEXT ] district_id [ INTEGER ] reviews.district_id = district.district_id External Knowledge: None\n-- Using valid SQLite and understading External Knowledge, answer the following questions for the tables provided above.\n-- What are the issues of the complains of male clients and products from age 25 and below?\nGenerate the SQL after thinking step by step: \nSELECT ' Output:SELECT DISTINCT T2.Issue FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T1.sex = 'Male' AND T1.age < 25;")
    demo_examples.append("Input:[INST] Here is a database schema: location : LocationID [ INTEGER ] primary_key Country [ TEXT ] State [ TEXT ] StateCode [ TEXT ] City [ TEXT ] user : UserID [ TEXT ] primary_key Gender [ TEXT ] twitter : TweetID [ TEXT ] primary_key Weekday [ TEXT ] Hour [ INTEGER ] Day [ INTEGER ] Lang [ TEXT ] IsReshare [ TEXT ] Reach [ INTEGER ] RetweetCount [ INTEGER ] Likes [ INTEGER ] Klout [ INTEGER ] Sentiment [ REAL ] text [ TEXT ] LocationID [ INTEGER ] twitter.LocationID = location.LocationID UserID [ TEXT ] twitter.UserID = user.UserID External Knowledge: None\n-- Using valid SQLite and understading External Knowledge, answer the following questions for the tables provided above. How many reshared tweets are there in Texas?\nGenerate the SQL after thinking step by step: \nSELECT ' Output:SELECT COUNT(T1.TweetID) FROM twitter AS T1 INNER JOIN location AS T2 ON T2.LocationID = T1.LocationID WHERE T2.State = 'Texas' AND T1.IsReshare = 'TRUE';")
    #pdb.set_trace()
    demo_cot_examples=["Input:[INST] Here is a database schema: lists : user_id [ INTEGER ] lists.user_id = lists_users.user_id list_id [ INTEGER ] primary_key list_title [ TEXT ] list_movie_number [ INTEGER ] list_update_timestamp_utc [ TEXT ] list_creation_timestamp_utc [ TEXT ] list_followers [ INTEGER ] list_url [ TEXT ] list_comments [ INTEGER ] list_description [ TEXT ] list_cover_image_url [ TEXT ] list_first_image_url [ TEXT ] list_second_image_url [ TEXT ] list_third_image_url [ TEXT ] movies : movie_id [ INTEGER ] primary_key movie_title [ TEXT ] movie_release_year [ INTEGER ] movie_url [ TEXT ] movie_title_language [ TEXT ] movie_popularity [ INTEGER ] movie_image_url [ TEXT ] director_id [ TEXT ] director_name [ TEXT ] director_url [ TEXT ] ratings_users : user_id [ INTEGER ] ratings_users.user_id = lists_users.user_id rating_date_utc [ TEXT ] user_trialist [ INTEGER ] user_subscriber [ INTEGER ] user_avatar_image_url [ TEXT ] user_cover_image_url [ TEXT ] user_eligible_for_trial [ INTEGER ] user_has_payment_method [ INTEGER ] lists_users : user_id [ INTEGER ] lists_users.user_id = lists.user_id list_id [ INTEGER ] lists_users.list_id = lists.list_id list_update_date_utc [ TEXT ] list_creation_date_utc [ TEXT ] user_trialist [ INTEGER ] user_subscriber [ INTEGER ] user_avatar_image_url [ TEXT ] user_cover_image_url [ TEXT ] user_eligible_for_trial [ TEXT ] user_has_payment_method [ TEXT ] ratings : movie_id [ INTEGER ] ratings.movie_id = movies.movie_id rating_id [ INTEGER ] rating_url [ TEXT ] rating_score [ INTEGER ] rating_timestamp_utc [ TEXT ] critic [ TEXT ] critic_likes [ INTEGER ] critic_comments [ INTEGER ] user_id [ INTEGER ] ratings.user_id = lists_users.user_id user_trialist [ INTEGER ] user_subscriber [ INTEGER ] user_eligible_for_trial [ INTEGER ] user_has_payment_method [ INTEGER ] External Knowledge: None\n-- Using valid SQLite and understading External Knowledge, answer the following questions for the tables provided above.\n-- Name movie titles released in year 1945. Sort the listing by the descending order of movie popularity?\nGenerate the SQL after thinking step by step: \nSELECT, Output:To generate the SQL query to get the movie titles released in the year 1945 and sort them by descending order of movie popularity, we need to follow these steps:\n\n1. **Identify the relevant table**: The table containing movie information is `movies`.\n2. **Identify the relevant columns**: We need the `movie_title`, `movie_release_year`, and `movie_popularity` columns.\n3. **Filter the data**: We need to filter the movies released in the year 1945.\n4. **Sort the data**: We need to sort the filtered movies by `movie_popularity` in descending order.\n5. **Select the required column**: We need to select the `movie_title` column.\n\nLet's break it down step by step:\n\n1. **From the `movies` table**:\n   ```sql\n   FROM movies\n   ```\n\n2. **Filter movies released in 1945**:\n   ```sql\n   WHERE movie_release_year = 1945\n   ```\n\n3. **Sort by movie popularity in descending order**:\n   ```sql\n   ORDER BY movie_popularity DESC\n   ```\n\n4. **Select the movie title**:\n   ```sql\n   SELECT movie_title\n. Therefore, we get the final SQL: SELECT movie_title FROM movies WHERE movie_release_year = 1945 ORDER BY movie_popularity DESC LIMIT 1;"]
    demo_cot_examples.append(["Input:[INST] Here is a database schema: authors : au_id [ TEXT ] primary_key au_lname [ TEXT ] au_fname [ TEXT ] phone [ TEXT ] address [ TEXT ] city [ TEXT ] state [ TEXT ] zip [ TEXT ] contract [ TEXT] jobs : job_id [ INTEGER ] primary_key job_desc [ TEXT ] min_lvl [ INTEGER ] max_lvl [ INTEGER ] publishers : pub_id [ TEXT ] primary_key pub_name [ TEXT ] city [ TEXT ] state [ TEXT ] country [ TEXT ] employee : emp_id [ TEXT ] primary_key fname [ TEXT ] minit [ TEXT ] lname [ TEXT ] job_id [ INTEGER ] employee.job_id = jobs.job_id job_lvl [ INTEGER ] pub_id [ TEXT ] employee.pub_id = publishers.pub_id hire_date [ DATETIME ] pub_info : pub_id [ TEXT ] primary_key pub_info.pub_id = publishers.pub_id logo [ BLOB ] pr_info [ TEXT ] stores : stor_id [ TEXT ] primary_key stor_name [ TEXT ] stor_address [ TEXT ] city [ TEXT ] state [ TEXT ] zip [ TEXT ] discounts : discounttype [ TEXT ] stor_id [ TEXT ] discounts.stor_id = stores.stor_id lowqty [ INTEGER ] highqty [ INTEGER ] discount [ REAL ] titles : title_id [ TEXT ] primary_key title [ TEXT ] type [ TEXT ] pub_id [ TEXT ] titles.pub_id = publishers.pub_id price [ REAL ] advance [ REAL ] royalty [ INTEGER ] ytd_sales [ INTEGER ] notes [ TEXT ] pubdate [ DATETIME ] roysched : title_id [ TEXT ] roysched.title_id = titles.title_id lorange [ INTEGER ] hirange [ INTEGER ] royalty [ INTEGER ] sales : stor_id [ TEXT ] sales.stor_id = stores.stor_id ord_num [ TEXT ] ord_date [ DATETIME ] qty [ INTEGER ] payterms [ TEXT ] title_id [ TEXT ] sales.title_id = titles.title_id titleauthor : au_id [ TEXT ] titleauthor.au_id = authors.au_id title_id [ TEXT ] titleauthor.title_id = titles.title_id au_ord [ INTEGER ] royaltyper [ INTEGER ] External Knowledge: None\n-- Using valid SQLite and understading External Knowledge, answer the following questions for the tables provided above.\n-- Please list the first names of the employees who work as Managing Editor. \nGenerate the SQL after thinking step by step: \nSELECT,Output:To generate the SQL query to list the first names of the employees who work as Managing Editor, we need to follow these steps:\n\n1. **Identify the relevant tables**: We need information from the `employee` and `jobs` tables.\n2. **Determine the necessary columns**: We need the `fname` column from the `employee` table and the `job_desc` column from the `jobs` table.\n3. **Establish the relationship**: The `employee` table has a `job_id` column that references the `job_id` column in the `jobs` table.\n4. **Filter the results**: We need to filter the results to only include employees whose job description is 'Managing Editor'.\n\nLet's break it down step by step:\n\n1. **Select the columns**: We need to select the `fname` column from the `employee` table.\n2. **Join the tables**: We need to join the `employee` table with the `jobs` table on the `job_id` column.\n3. **Apply the filter**: We need to filter the results where the `job_desc` in the `jobs` table is 'Managing Editor'. Therefore, we get the final SQL: SELECT T1.fname FROM employee AS T1 INNER JOIN jobs AS T2 ON T1.job_id = T2.job_id WHERE T2.job_desc = 'Managing Editor';"])
    demo_cot_examples.append(["Input:[INST] Here is a database schema: state : StateCode [ TEXT ] primary_key State [ TEXT ] Region [ TEXT ] callcenterlogs : Date received [ DATE ] Complaint ID [ TEXT ] primary_key rand client [ TEXT] callcenterlogs.rand client = client.client_id phonefinal [ TEXT ] vru+line [ TEXT ] call_id [ INTEGER ] priority [ INTEGER ] type [ TEXT ] outcome [ TEXT ] server [ TEXT ] ser_start [ TEXT ] ser_exit [ TEXT ] ser_time [ TEXT ] client : client_id [ TEXT ] primary_key sex [ TEXT ] day [ INTEGER ] month [ INTEGER ] year [ INTEGER ] age [ INTEGER ] social [ TEXT ] first [ TEXT ] middle [ TEXT ] last [ TEXT ] phone [ TEXT ] email [ TEXT ] address_1 [ TEXT ] address_2 [ TEXT ] city [ TEXT ] state [ TEXT ] zipcode [ INTEGER ] district_id [ INTEGER ] client.district_id = district.district_id district : district_id [ INTEGER ] primary_key city [ TEXT ] state_abbrev [ TEXT ] district.state_abbrev = state.StateCode division [ TEXT ] events : Date received [ DATE ] Product [ TEXT ] Sub-product [ TEXT ] Issue [ TEXT ] Sub-issue [ TEXT ] Consumer complaint narrative [ TEXT ] Tags [ TEXT ] Consumer consent provided? [ TEXT ] Submitted via [ TEXT ] Date sent to company [ TEXT ] Company response to consumer [ TEXT ] Timely response? [ TEXT ] Consumer disputed? [ TEXT ] Complaint ID [ TEXT ] events.Complaint ID = callcenterlogs.Complaint ID Client_ID [ TEXT ] events.Client_ID = client.client_id reviews : Date [ DATE ] primary_key Stars [ INTEGER ] Reviews [ TEXT ] Product [ TEXT ] district_id [ INTEGER ] reviews.district_id = district.district_id External Knowledge: None\n-- Using valid SQLite and understading External Knowledge, answer the following questions for the tables provided above.\n-- What are the issues of the complains of male clients and products from age 25 and below?\nGenerate the SQL after thinking step by step: \nSELECT, Output:To generate the SQL query to find the issues of complaints from male clients aged 25 and below, we need to follow these steps:\n\n1. **Identify the relevant tables**: We need to use the `client` and `events` tables.\n2. **Determine the necessary columns**: We need the `sex` and `age` columns from the `client` table and the `Issue` column from the `events` table.\n3. **Establish the relationships**: The `client` table is related to the `events` table through the `client_id` and `Client_ID` columns.\n4. **Set the conditions**: We need to filter for male clients (`sex = 'Male'`) and those aged 25 and below (`age < 25`).\n5. **Select the required output**: We need to select distinct issues from the `events` table.\n\nLet's break down the SQL query step by step:\n\n1. **Select the columns**: We need to select the `Issue` column from the `events` table.\n2. **Join the tables**: We need to join the `client` and `events` tables on the `client_id` and `Client_ID` columns.\n3. **Apply the conditions**: We need to filter the results where `sex` is 'Male' and `age` is less than 25.\n4. **Ensure distinct results**: Use `DISTINCT` to avoid duplicate issues. Therefore, we get the final SQL: SELECT DISTINCT T2.Issue FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T1.sex = 'Male' AND T1.age < 25;"])
    demo_cot_examples.append(["Input:[INST] Here is a database schema: location : LocationID [ INTEGER ] primary_key Country [ TEXT ] State [ TEXT ] StateCode [ TEXT ] City [ TEXT ] user : UserID [ TEXT ] primary_key Gender [ TEXT ] twitter : TweetID [ TEXT ] primary_key Weekday [ TEXT ] Hour [ INTEGER ] Day [ INTEGER ] Lang [ TEXT ] IsReshare [ TEXT ] Reach [ INTEGER ] RetweetCount [ INTEGER ] Likes [ INTEGER ] Klout [ INTEGER ] Sentiment [ REAL ] text [ TEXT ] LocationID [ INTEGER ] twitter.LocationID = location.LocationID UserID [ TEXT ] twitter.UserID = user.UserID External Knowledge: None\n-- Using valid SQLite and understading External Knowledge, answer the following questions for the tables provided above. How many reshared tweets are there in Texas?\nGenerate the SQL after thinking step by step: \nSELECT, Output:To generate the SQL query to find the number of reshared tweets in Texas, we need to follow these steps:\n\n1. **Understand the Schema**: \n   - The `twitter` table contains information about tweets, including whether they are reshared (`IsReshare`).\n   - The `location` table contains information about locations, including the state.\n   - The `twitter` table has a foreign key `LocationID` that references the `LocationID` in the `location` table.\n\n2. **Identify the Relevant Tables and Columns**:\n   - We need to count the number of tweets (`TweetID`) from the `twitter` table.\n   - We need to filter these tweets based on the state being 'Texas' from the `location` table.\n   - We need to filter tweets that are reshared (`IsReshare` = 'TRUE').\n\n3. **Formulate the SQL Query**:\n   - We will use an `INNER JOIN` to combine the `twitter` and `location` tables based on the `LocationID`.\n   - We will use a `WHERE` clause to filter the results to only include tweets from Texas and those that are reshared.\n\n4. **Construct the SQL Query**:\n   - Select the count of `TweetID` from the `twitter` table.\n   - Join the `twitter` table with the `location` table on `LocationID`.\n   - Filter the results where the state is 'Texas' and `IsReshare` is 'TRUE'. Therefore, we get the final SQL: SELECT COUNT(T1.TweetID) FROM twitter AS T1 INNER JOIN location AS T2 ON T2.LocationID = T1.LocationID WHERE T2.State = 'Texas' AND T1.IsReshare = 'TRUE';"])    
    too_many_keys=[]
    if not demos:
        demo_cot_examples=torch.load('valid_data_cot.pt')
    else:
        demo_cot_examples=demos
    #pdb.set_trace() #input/output format
    prompt="""The task is '{0}', you can induce the its input and output format pattern in general from these examples:{1}. You only need to output the format pattern:""".format(instruction,demo_examples) #high_quality_$
    pattern=query_azure_openai_chatgpt_chat(prompt)
    #necessary
#    prompt="""Based on the task instruction:'{0}' and the example format:'{1}', you can generate necessary conditions about training examples of this task. Necessary conditions are basic rules that all
#examples of this task must satisfy them. Conditions:""".format(instruction,pattern)
#    necessary_conditions=query_azure_openai_chatgpt_chat(prompt)
    #pdb.set_trace()
    #task knowledge
#    task_knowledge_prompt="""Based on the task instruction:'{0}', you can generate general task knowledge about this task. Details for processing various situations in the task are perferred. Don't give me specific example. Task knowledge:""".format(instruction)
#    task_knowledge=query_azure_openai_chatgpt_chat(task_knowledge_prompt)
#    task_instruction+=' '+task_knowledge  #domain knowledge
#    domain_prompt="You can write some general description of this domain/topic:{0}. Your output should only contain the description and be short.".format(domain)
#    domain_knowledge=query_azure_openai_chatgpt_chat(domain_prompt)
    #instruction=domain_knowledge+'\n'+instruction
    #reading process
    #procedure_prompt="You are an expert for reading passages and generating task data of task {0}. You are going to design the procedure in general for how to read the passage related to {1} and extract its knowledge to construct the data. No example is needed. You only output the procedure. The knowledge about the domain in general is {2}".format(instruction,domain,domain_knowledge) #    pdb.set_trace()
    #procedure=query_azure_openai_chatgpt_chat(procedure_prompt)
    #high_scores_rules
#    prompt="""Based on the knowledge of domain {0} and task {1}. You can write evaluation rules for being high quality samples of this tasks. Examples satisfied with these rules are better than other ex
#amples. No examples are needed. Rules in list:""".format(domain_knowledge, task_knowledge)
#    high_rules=query_azure_openai_chatgpt_chat(prompt)
    domain_knowledge=''    #exceptions
#    prompt="""Based on the knowledge of task {0} and domain {1}. You can write exceptions and errors in low quality samples of this task in general. These exceptions and errors cause the need of revising the samples. No examples are needed. Exceptions and errors in list:""".format(task_knowledge, domain_knowledge)
#    exceptions=query_azure_openai_chatgpt_chat(prompt)
    #transform task knowledge into data generation instruction
    generation_prompt="You are an expert for generating data. You are going to design data generation strategies and instructions to guide high-quality,diverse,and challenging data generation of the task '{0}'. You should directly output the data generation strategies and instructions".format(instruction)
#    pdb.set_trace() #transform the necessary conditions, high_score_ruls, and exceptions into data evaluation principles
#    prompt="""You are a data evaluator responsible for assigning scores to the quality of samples on a scale of 0 to 5, with higher scores indicating better quality (e.g., sample diffculty,language precision). The sample is associated with the task: {0}. You can establish detailed scoring criteria for each score level based on your knowledge specific content from the following sources: high score rules:'{1}'. Your output should follow this python list format: ['Score 5: xxx', 'Score 4: xxx', ...].""".format(instruction,high_rules) 
#    evaluation_criteria=query_azure_openai_chatgpt_chat(prompt) #pdb.set_trace()
    
#    pdb.set_trace()
    try:
        examples_ori=torch.load('{0}.pt'.format(store_name))
    except:
        examples_ori=[]
    try:
        examples_ori_all=torch.load('{0}_all.pt'.format(store_name))
    except:
        examples_ori_all=[]
    #types=["Complex Join Operations**: The model struggles with queries that require multiple table joins to retrieve the correct data, especially when the joins involve foreign key relationships and specific conditions across different tables.","**Nested Queries and Subqueries**: The model has difficulty generating correct SQL for nested queries and subqueries, particularly when the nested query involves conditions that need to be matched with the outer query.","**Specific Column Selection and Conditions**: The model often fails to select the correct columns or apply the correct conditions to columns,leading to incomplete or incorrect SQL statements. This includes issues with selecting columns that do not exist or misapplying conditions.","**Handling of Missing or Implicit Information**: The model struggles when the query requires understanding or inferring missing or implicit information, such as default values or implicit relationships between tables.","**Aggregation Functions and Grouping**: The model has issues correctly applying aggregation functions (e.g., SUM, MAX, COUNT) in the context of complex queries, especially when these functions need to be combined with other SQL operations such as grouping and ordering.","**Date and Time Handling**: The model has difficulty correctly interpreting and using date and time formats, which can lead to errors in queries that involve date comparisons or date-based conditions."] #    instruction=query_azure_openai_chatgpt_chat(generation_prompt)
#    pdb.set_trace() #generation_instruction=eval(generation_instruction) #pdb.set_trace()#basic necessary conditions:'{1}', and exceptions and errors:'{3}'
#    instruction=domain_knowledge+'\n'+instruction
#    instruction+='\nYou can refer to these data generation instructions:{0}'.format(generation_instruction)
#    pdb.set_trace()
    start=len(examples_ori)
    for num_id in range(start,num):
        '''
        if len(generation_instruction)>3:
            compound_instruction=instruction+'\nYou can refer to these data generation instructions:{0}'.format(random.sample(generation_instruction,3)) #print(num_id)
        else:
            compound_instruction=instruction+'\nYou can refer to these data generation instructions:{0}'.format(generation_instruction)
        '''
        #import random
#        random.seed(id)
        sample_size = 1 # random.randint(1,5)
        sampled_objects = random.sample(passages, sample_size)
#        pdb.set_trace()
        errors=0
        #examples_ori=[]
        if generated_examples:
#            pdb.set_trace()
            best_example=None #compound_instruction=instruction+'\nYou can refer to these data generation instructions:{0}'.format(random.choice(generation_instruction,3))
            for _ in range(1):
                new_example=aug_few_shot_prompt_generate(demo_examples,instruction,task_instruction,too_many_keys,sampled_objects, pattern,domain_knowledge,previous_examples=generated_examples)
                #pdb.set_trace()
                evaluation_criteria="### General Criteria for Judging Data Quality (0~5 Score)\n\n#### 0-1 Score: Low Quality Data\n1. **Incorrect Actions**: The predicted actions do not align with the required actions specified in the task.\n2. **Irrelevant Actions**: The predictions include actions that are not relevant to the task.\n3. **Missing Key Actions**: Essential actions required to complete the task are absent.\n4. **Incorrect Object Interactions**: The predictions involve incorrect or unnecessary interactions with objects or locations.\n5. **Incorrect Sequence of Actions**: The actions are not logically ordered, disrupting task completion.\n6. **Overly Complex or Redundant Actions**: The actions are unnecessarily complex or repetitive.\n7. **Lack of Context Understanding**: The predictions show a lack of understanding of the task's context and environment.\n\n#### 2-3 Score: Medium Quality Data\n1. **Minor Incorrect Actions**: Some actions do not align with the task but do not significantly hinder task completion.\n2. **Few Redundant or Extraneous Actions**: There are a few unnecessary or repetitive actions, but they do not majorly impact the task.\n3. **Occasional Irrelevant Actions**: Some actions are unrelated to the task but are not frequent.\n4. **Some Missing Key Actions**: A few essential actions are missing, but the task can still be partially completed.\n5. **Minor Incorrect Object Interactions**: Some interactions with objects or locations are not entirely logical but do not severely disrupt the task.\n6. **Minor Sequence Errors**: The order of actions has minor errors that do not critically affect task completion.\n7. **Partial Context Understanding**: The predictions show a partial understanding of the task's context, with some inappropriate actions.\n\n#### 4-5 Score: High Quality Data\n1. **Correct Actions**: The predicted actions align well with the required actions to complete the task.\n2. **Minimal Redundant or Extraneous Actions**: There are very few or no unnecessary or repetitive actions.\n3. **Relevant Actions**: All actions are related to the task at hand.\n4. **Complete Key Actions**: All essential actions required to complete the task are present.\n5. **Correct Object Interactions**: Interactions with objects and locations are logical and feasible.\n6. **Correct Sequence**: The order of actions is logical and facilitates task completion.\n7. **Good Context Understanding**: The predictions show a good understanding of the task's context and environment, leading to appropriate actions."
                score=evaluate_score_reason(str(new_example),evaluation_criteria)
                pdb.set_trace()
                examples_ori.append(new_example)
                '''
                try:
    #                new_example=extract_examples(new_example)
                    max_score=0
                    #max_index=0
                    #for example_id, example in enumerate(examples): 
#                    pdb.set_trace()
                    score=evaluate_score_reason(new_example,evaluation_criteria) #query_azure_openai_chatgpt_chat(template)
                    match = re.findall(r'\b[0-5]\b', score)
                    number=int(match[0])if match else None
                    max_num=number #pdb.set_trace()
                    best_example=new_example
                    if number and number!=5:
                        for _ in range(5): #pdb.set_trace()
                            new_example=aug_few_shot_prompt_generate(demo_examples,instruction,too_many_keys,sampled_objects, pattern,domain_knowledge,revision='Example: '+str(new_example)+' Analysis:'+score,previous_examples=generated_examples)
                            new_example=new_example[new_example.find('{'):new_example.rfind('}')+1] #new_example=extract_examples(new_example)
                            score=evaluate_score_reason(new_example,evaluation_criteria) #query_azure_openai_chatgpt_chat(template)
                            match = re.findall(r'\b[0-5]\b', score)
                            number=int(match[0])if match else None
                            best_example=new_example if number>=max_num else best_example
                            max_num=max(max_num,number)
#                            pdb.set_trace()
                            if number==5:
                                break
                        #pdb.set_trace()
                    #if score and score>=max_score:
                         #if score>=max_score:
                    #         max_score=score
                    #best_example=new_example
                #generated_examples.append(examples[max_index])
                         #scores.append(score) #generated_examples.append((example,score))
                except:
                    #pdb.set_trace()
                    errors+=1
            if best_example:
                generated_examples.append(best_example)
            '''
        else:
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
            best_example=None
            for _ in range(1):
                op_instruction=random.sample(operation_instructions,1)
                re_instruction=random.choices(revise_instruction,k=3)
                r_examples=random.sample(demo_cot_examples, sample_num)
                if types:
                    choiced_type=random.sample(types,1)
                    data_instruction=instruction+'You should focus on generating data with thess characteristic/type:{0}'.format(choiced_type)
                else:
                    data_instruction=instruction
                print(data_instruction,'\n') #,op_instruction,'\n')
                all_example,new_example=aug_few_shot_prompt_generate(r_examples,data_instruction,task_instruction,too_many_keys,sampled_objects,pattern,domain_knowledge,operation_instruction=op_instruction,revise_instruction=re_instruction)
#                pdb.set_trace()
                if new_example: #pdb.set_trace()
                    #examples_ori_all.extend(all_example)
                    examples_ori.append(new_example)
                    #torch.save(examples_ori_all,'{0}_all.pt'.format(store_name))
                    torch.save(examples_ori,'{0}.pt'.format(store_name))
                if all_example:
                    examples_ori_all.extend(all_example)
                    torch.save(examples_ori_all,'{0}_all.pt'.format(store_name))
                '''
                try:
                    
                    #new_example=extract_examples(new_example)
                    max_score=0
                    #max_index=0
                    #pdb.set_trace() #for example_id, example in enumerate(examples): 
                    score=evaluate_score_reason(new_example,evaluation_criteria) #query_azure_openai_chatgpt_chat(template)
                    match = re.findall(r'\b[0-5]\b', score)
                    number=int(match[0])if match else None
                    max_num=number 
                    best_example=new_example #pdb.set_trace()
                    if number and number!=5:
                        for _ in range(5): #pdb.set_trace()
                            new_example=aug_few_shot_prompt_generate(demo_examples,instruction,too_many_keys,sampled_objects, pattern,domain_knowledge,revision='Example: '+str(new_example)+' Analysis:'+score,previous_examples=generated_examples)
                            #pdb.set_trace()
                            new_example=new_example[new_example.find('{'):new_example.rfind('}')+1]
                            score=evaluate_score_reason(new_example,evaluation_criteria) #query_azure_openai_chatgpt_chat(template)
                            match = re.findall(r'\b[0-5]\b', score)
                            number=int(match[0])if match else None
                            best_example=new_example if number>=max_num else best_example
                            max_num=max(max_num,number)
                            if number==5:
                                break
                            #pdb.set_trace() #if number and number!=5:
                    
                    #    pdb.set_trace()
                    #best_example=new_example #if score and number>=max_score:
                         #if score>=max_score:
                    #         max_score=score
                    #         best_example=new_example
                #generated_examples.append(examples[max_index])
                         #scores.append(score) #generated_examples.append((example,score))
                except:
                              
                    #pdb.set_trace()
                    errors+=1
            #pdb.set_trace()
            if best_example:
                generated_examples.append(best_example[0])
            #generated_examples.append(new_example)
            '''
            #pdb.set_trace()
        #try:
        #    examples.extend(multiple_examples_extraction(new_example))
        #except:
        #    continue
        if examples_ori:
            print(examples_ori[-1])
        print(num_id,len(examples_ori_all),len(examples_ori),errors)
        '''
        try:
            new_example=eval(new_example[new_example.find('{'):new_example.find('}')+1])
            output=new_example['output']
        except:
            input_output=str(new_example).split('input')[1]
            if 'output' in input_output:
                input,output=input_output.split('output')[0],input_output.split('output')[1]
            else:
                input,output=input_output.split('Output')[0],input_output.split('Output')[1]
            new_example={'input':input.replace(':','').replace('"','').replace('\n','').strip(),'output':output.replace(':','').replace('"','').replace('}','').replace('\n','').strip()} 
        if new_example['output'] in output_count.keys():
            output_count[new_example['output']]+=1
        else:
            output_count[new_example['output']]=1
        too_many_keys=keys_above_average(output_count)
        '''
    #examples_num={'neutral':0,'entailment':0,'contradiction':0} #['neutral', 'entailment', 'contradiction']
    #generated_examples=sorted(generated_examples,key=lambda x:x[1],reverse=True)
    #sorted_examples=[]
    #for example in generated_examples:
    #    if examples_num[example[0]['output'].lower().replace('output','').strip()]>(num//len(['neutral', 'entailment', 'contradiction'])):
    #        continue
    #    else:
    #        sorted_examples.append(example[0])
    #        examples_num[example[0]['output'].lower().replace('output','').strip()]+=1
     #    pdb.set_trace()  #   generated_examples.append(new_example)
    return examples_ori_all,examples_ori
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
        if 'input' in split:
            raw_examples.append(split)
    examples=[]
    for raw_example in raw_examples:
        input_tag='"input":' if '"input":' in str(raw_example) else "'input':" #try:
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
        new_example={'input':input.replace(':','').replace('{','').replace('"','').replace('\n','').strip(),'output':output.replace(':','').replace('"','').replace('}','').replace('\n','').strip()}
        examples.append(new_example)
    return examples
