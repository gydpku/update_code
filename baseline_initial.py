from auto_finetune import train_model,train_status_check
import numpy as np
from ooa_instruction_optimization_constrast_t_3 import run_ooa_instruction_optimization #,run_ooa_instruction_optimization_loop
task_instruction="The task is to generate medical inference data based on the provided medical passage."
from valid_data_analysis import data_type_analysis
from datasets import load_dataset
from ppl_calculation import sort_by_perplexity
from model_soups_method import find_best_combination
from bayesian_soup_search import bayesian_search
import torch
import re
from openai_call import query_azure_openai_chatgpt_chat
import pdb
from datasets import concatenate_datasets
from datasets import load_from_disk
from generate_data import clean_and_collect_dataset
from model_eval import process_validation_batch_major,valid_results_collect,process_validation_batch_count
import logging
import os
import shutil
experiment_name='random_3k_baseline_median_max_gradient_group_initial_data_initial'
output_path='/dccstor/obsidian_llm/yiduo/copy_v2/finetuned_models/'
train_seed=2024
test_examples=[]
task_name='MedNLI'
dataset=load_dataset('hippocrates/MedNLI_test')  
for id in range(len(dataset['test'])):test_examples.append({'Input':dataset['test'][id]['query'],'Output':dataset['test'][id]['answer']})
valid_data=[] 
for id in range(len(dataset['train'])):valid_data.append({'Input':dataset['train'][id]['query'],'Output':dataset['train'][id]['answer']}) #torch.load('{0}_demo.pt'.format(args.task))
valid_data=valid_data[:100]
valid_data=torch.load('nli_demo.pt')
initial_model_path='/dccstor/obsidian_llm/yiduo/copy_v2/finetuned_models/randoms_data_3k_model'
initial_data_path='naive_3k_random'
# Configure logging
logging.basicConfig(
    filename="{0}.log".format(experiment_name), 
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)
def remove_previous_folders(experiment_name,iteration):
    if iteration>0:
        prev_names=torch.load('{0}_names.pt'.format(experiment_name+'_'+str(iteration-1)))
        prev_model_paths=[os.path.join(output_path, model_name+'_model') for model_name in prev_names]
        global_best_path=torch.load('global_best_model_path_{0}_{1}.pt'.format(experiment_name,iteration-1))
        prev_best_path=torch.load('cur_best_path_{0}_{1}.pt'.format(experiment_name,iteration-1))
        for path in prev_model_paths:
            if path!=prev_best_path and path!=global_best_path:
                remove_folder(path)
def remove_folder(path):
    if os.path.isdir(path):  # Check if the directory exists
        shutil.rmtree(path)
        print(f"Directory '{path}' has been removed.")
    else:
        print(f"Directory '{path}' does not exist.")
for iteration in range(1,5):
    print('Generating data for iteration ',iteration)
    if iteration==0:
        global_best_model_path=initial_model_path
        global_best_dataset_path=initial_data_path
        global_best_group=[global_best_model_path]
        previous_gradients=None
        try:
            ooa_data=torch.load('ooa_failed_cases_{0}.pt'.format(experiment_name))+torch.load('im_failed_cases_{0}.pt'.format(experiment_name))
        except:
            ooa_failed_cases, im_failed_cases, correct_cases=process_validation_batch_major(global_best_model_path, valid_data,task=task_name,seed=False, iteration=100)  
            torch.save(ooa_failed_cases,'ooa_failed_cases_{0}.pt'.format(experiment_name))
            torch.save(im_failed_cases,'im_failed_cases_{0}.pt'.format(experiment_name))
            ooa_data=torch.load('ooa_failed_cases_{0}.pt'.format(experiment_name))+torch.load('im_failed_cases_{0}.pt'.format(experiment_name))
        global_best_performance=1-len(ooa_data)/(len(valid_data))
        instructions_data=run_ooa_instruction_optimization(ooa_data,task_instruction,'','nli',experiment_name+'_'+str(iteration),'Medical',data_num=20)
    else:
        ooa_data=torch.load('ooa_failed_cases_{0}_{1}.pt'.format(experiment_name,iteration-1))+torch.load('im_failed_cases_{0}_{1}.pt'.format(experiment_name,iteration-1))
        previous_gradients=[]
        global_best_dataset_path=torch.load('global_best_dataset_path_{0}_{1}.pt'.format(experiment_name,iteration-1))
        for i in range(iteration):
            previous_gradient_path='chosen_gradient_{0}_{1}.pt'.format(experiment_name,i)
            previous_gradients.append(torch.load(previous_gradient_path))
#        pdb.set_trace()
        global_best_model_path=torch.load('global_best_model_path_{0}_{1}.pt'.format(experiment_name,iteration-1))
        global_best_performance=torch.load('global_best_performance_{0}_{1}.pt'.format(experiment_name,iteration-1))
        global_best_group=torch.load('global_best_group_{0}_{1}.pt'.format(experiment_name,iteration-1))

        #global_best_data_name=torch.load('global_best_data_name_{0}_{1}.pt'.format(experiment_name,iteration-1))
        instructions_data=run_ooa_instruction_optimization(ooa_data,task_instruction,'','nli',experiment_name+'_'+str(iteration),'Medical',data_num=20,previous_gradients=previous_gradients)
     #    pdb.set_trace()
    keys=[key for key in instructions_data.keys()]
    global_best_data=load_from_disk(global_best_dataset_path)
    #global_filter
    def transform_from_low_to_up(data):
            new_data=[]
            for item in data:
                new_data.append({'Input':item['instruction'],'Output':item['output']})
            return new_data
    global_best_data=transform_from_low_to_up(global_best_data)
    global_data_count=process_validation_batch_count(initial_model_path, global_best_data,task=task_name,seed=False, iteration=10)
    scores=[item[1] for item in global_data_count]
    median=np.median(scores)
    global_data_count.sort(key=lambda x:abs(x[1]-median))
    global_best_data=[item[0] for item in global_data_count]
    print('Constructing training data for iteration ',iteration)
    names_keys={}
    try:
        names=torch.load('{0}_names.pt'.format(experiment_name+'_'+str(iteration)))
        names_valid=torch.load('{0}_names_valid.pt'.format(experiment_name+'_'+str(iteration)))
        names_dataset=torch.load('{0}_names_dataset.pt'.format(experiment_name+'_'+str(iteration)))
        names_keys=torch.load('{0}_names_keys.pt'.format(experiment_name+'_'+str(iteration)))

    except:
        names=[]
        names_valid={}
        names_dataset={}
        for key in keys:
            examples=instructions_data[key][2]
            data=[example for example in examples if example]
            def transform_from_low_to_up(data):
                new_data=[]
                for item in data:
                    new_data.append({'Input':item['input'],'Output':item['output']})
                return new_data
            def transform_from_up_to_low(data):
                new_data=[]
                for item in data:
                    new_data.append({'input':item['Input'],'output':item['Output']})
                return new_data

            data=transform_from_low_to_up(data)
            data_count=process_validation_batch_count(initial_model_path, data,task=task_name,seed=False, iteration=100)
            scores=[item[1] for item in data_count]
            median=np.median(scores)
            data_count.sort(key=lambda x:abs(x[1]-median))
            data=[item[0] for item in data_count]
            print(key,' scores are ',scores)
            data=transform_from_up_to_low(data)
            #data = sort_by_perplexity(data, global_best_model_path)
            #data = [item[1] for item in sorted(data, key=lambda x:x[0])]
            print(len(examples),len(data))
            prompt="make a name (one word) for this characteristic, directly and only output the name:{0}".format(key)
            name=query_azure_openai_chatgpt_chat(prompt)
            names_keys[name]=key
            names_valid[name]=instructions_data[key][1]
            for num in [100,300,500]:
                cur_data=transform_from_up_to_low(global_best_data[:-num])
                cur_data.extend(data[:num])
                #cur_data=transform_from_up_to_low(cur_data)
                clean_and_collect_dataset(cur_data,'',experiment_name+name+'_'+str(iteration)+'_'+str(num))
                #key_dataset=load_from_disk(name)
                #dataset_concat=concatenate_datasets([global_best_data,key_dataset])
                #dataset_concat.save_to_disk(experiment_name+name+'_'+str(iteration)+'_'+str(num))
                print(experiment_name+name+'_'+str(iteration)+'_'+str(num))
                names.append(experiment_name.replace('random_3k_','')+name+'_'+str(iteration)+'_'+str(num))
                names_dataset[name+'_'+str(iteration)+'_'+str(num)]=experiment_name+name+'_'+str(iteration)+'_'+str(num)
        print("Now train your model on these datasets and collect the validation results,save their names! as {0}_names.pt".format(experiment_name+'_'+str(iteration)))
        torch.save(names,'{0}_names.pt'.format(experiment_name+'_'+str(iteration)))
        torch.save(names_dataset,'{0}_names_dataset.pt'.format(experiment_name+'_'+str(iteration)))
        torch.save(names_valid,'{0}_names_valid.pt'.format(experiment_name+'_'+str(iteration)))
        torch.save(names_keys,'{0}_names_keys.pt'.format(experiment_name+'_'+str(iteration)))
    print('Training models for iteration ',iteration)
    for name in names:
        output_path='/dccstor/obsidian_llm/yiduo/copy_v2/finetuned_models/'
        cur_path='/dccstor/obsidian_llm/yiduo/summary/src/'
        base_model_path='/dccstor/obsidian_llm/yiduo/h100_data/llama-3-8b'
        model_output_path=os.path.join(output_path,name+'_model')
        failed_times=0
        while not train_status_check(model_output_path):
            train_model(os.path.join(cur_path,'random_3k_'+name),base_model_path,model_output_path,seed=train_seed)
            if not train_status_check(model_output_path):
                remove_folder(model_output_path)
                failed_times+=1
                print('We have failed',failed_times,'times')
                if failed_times>3:
                    pdb.set_trace()
        print('We have finished training the model ',model_output_path)
    print('Selecting the best model by model soup search for iteration ',iteration)
    try:
        global_best_model_path=torch.load('global_best_model_path_{0}_{1}.pt'.format(experiment_name,iteration))
        cur_best_path=torch.load('cur_best_path_{0}_{1}.pt'.format(experiment_name,iteration))
    except:
        #global_best_model_path

        model_paths = [os.path.join(output_path, model_name+'_model') for model_name in names]
        dataset_names=list(set([name.split('_')[0] for name in names_dataset.keys()]))
        model_groups={}
        
        for dataset_name in dataset_names:
            model_group=[model_path for model_path in model_paths if dataset_name in model_path]
            model_group.extend(global_best_group)
            model_groups[dataset_name]=model_group  
        group_best_paths_acc=[]
        for dataset_name in model_groups.keys():
            model_group=model_groups[dataset_name]
            best_path,best_performance=bayesian_search(model_group,exp_name=experiment_name,task=task_name)
            group_best_paths_acc.append((best_path,best_performance,dataset_name))
        group_best_paths_acc.sort(key=lambda x:x[1],reverse=True)
        cur_best_model_path=group_best_paths_acc[0][0]
        cur_best_performance=group_best_paths_acc[0][1]
        cur_best_dataset_name=group_best_paths_acc[0][2]
        cur_best_dataset_path=names_dataset[sorted(list(set([name for name in names_dataset.keys() if cur_best_dataset_name in name])),key=lambda x: int(re.search(r'_(\d+)$', x).group(1)),reverse=True)[0]]
        cur_best_group=model_groups[cur_best_dataset_name]
        for path in group_best_paths_acc[1:]:
            remove_folder(path[0])


        #best_path,best_performance=bayesian_search(model_paths,task='nli',m_voting_num=10)
        if cur_best_performance>global_best_performance:
            global_best_model_path=cur_best_model_path
            global_best_performance=cur_best_performance
            print('We have found a better model!')
            f_test,c_test=valid_results_collect(global_best_model_path, test_examples, task_name)
            avg_test_acc=len(c_test)/(len(c_test)+len(f_test))
            print(len(c_test)/(len(c_test)+len(f_test)),'avg_acc')
            chosen_keys=names_keys[cur_best_dataset_name]
            print('The chosen gradient is',chosen_keys)
            global_best_group=cur_best_group
            global_best_dataset_path=cur_best_dataset_path
            ooa_failed_cases, im_failed_cases, correct_cases=process_validation_batch_major(global_best_model_path, valid_data,task=task_name,seed=False, iteration=100)  
        else:
            chosen_keys=''
            
        torch.save(global_best_model_path,'global_best_model_path_{0}_{1}.pt'.format(experiment_name,iteration))
        torch.save(global_best_model_path,'cur_best_path_{0}_{1}.pt'.format(experiment_name,iteration))
        torch.save(chosen_keys,'chosen_gradient_{0}_{1}.pt'.format(experiment_name,iteration))
        torch.save(ooa_failed_cases,'ooa_failed_cases_{0}_{1}.pt'.format(experiment_name,iteration))
        torch.save(im_failed_cases,'im_failed_cases_{0}_{1}.pt'.format(experiment_name,iteration))
        torch.save(global_best_group,'global_best_group_{0}_{1}.pt'.format(experiment_name,iteration))
        torch.save(global_best_performance,'global_best_performance_{0}_{1}.pt'.format(experiment_name,iteration))
        torch.save(global_best_dataset_path,'global_best_dataset_path_{0}_{1}.pt'.format(experiment_name,iteration))
        
        logging.info("Logging important variables:")
        logging.info(f"Iteration {iteration}: global_best_model_path = {global_best_model_path}")
#        logging.info(f"Iteration {iteration}: global_best_data_name = {global_best_data_name}")
        logging.info(f"Iteration {iteration}: names = {names}")
        logging.info(f"Iteration {iteration}: names_dataset = {names_dataset}")
        logging.info(f"Iteration {iteration}: best_performance = {best_performance}")
        logging.info(f"Iteration {iteration}: avg_test_acc = {avg_test_acc}")
   
