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
experiment_name='instruct_baseline_gsm8k_3.1_doc_rate_0.1' #'instruct_baseline_gsm8k_3.1_rate_0.1'
output_path='/dccstor/obsidian_llm/yiduo/copy_v2/finetuned_models/'
cur_path='/dccstor/obsidian_llm/yiduo/summary/src/'
rate=0.3
model='LLama3-8B'
base_model_path='/dccstor/obsidian_llm/yiduo/Llama-3.1-8B-Instruct' #'/dccstor/obsidian_llm/yiduo/llama-3-instruct' #'/dccstor/obsidian_llm/yiduo/h100_data/llama-3-8b'
task_name='GSM8k'
train_seed=2024
task_instruction='You are given a word problem involving basic arithmetic, algebra, or geometry. Your task is to carefully read the problem and provide a step-by-step solution, ensuring that all intermediate steps are shown clearly.'
#LogiQAbaseline_logiqa
def load_task_dataset(dataset_name,valid_num=100):
    if dataset_name=='MedNLI':
        test_examples=[]
        dataset=load_dataset('hippocrates/MedNLI_test')  
        for id in range(len(dataset['test'])):test_examples.append({'Input':dataset['test'][id]['query'],'Output':dataset['test'][id]['answer']})
        valid_data=[] 
        for id in range(len(dataset['train'])):valid_data.append({'Input':dataset['train'][id]['query'],'Output':dataset['train'][id]['answer']}) #torch.load('{0}_demo.pt'.format(args.task))
        valid_data=valid_data[:valid_num]
        domain='Medical'
    if dataset_name=='GSM8k':
        test_examples=[]
        print('dataset downloading')
        dataset=load_dataset('openai/gsm8k','main',cache_dir='/dccstor/obsidian_llm/yiduo')
        print('valid and train split')
        for id in range(len(dataset['test'])):test_examples.append({'Input':dataset['test'][id]['question'],'Output':dataset['test'][id]['answer']})
        valid_data=[] 
        for id in range(len(dataset['train'])):valid_data.append({'Input':dataset['train'][id]['question'],'Output':dataset['train'][id]['answer']}) #torch.load('{0}_demo.pt'.format(args.task))
        valid_data=valid_data[:valid_num]
        domain='Math,Arithmetic'
    if dataset_name=='LogiQA':
        test_examples=[]
        dataset=load_dataset('lucasmccabe/logiqa',cache_dir='/dccstor/obsidian_llm/yiduo')
        def transform_options(options):
            options_str=''
            for option_id,option in enumerate(options):
                options_str+='{0}:{1}\n'.format(chr(ord('A')+option_id),option)
            return options_str
        for id in range(len(dataset['test'])):test_examples.append({'Input':'Context:{0}\nQuestion:{1}\nOptions:{2}'.format(dataset['test'][id]['context'],dataset['test'][id]['query'],transform_options(dataset['test'][id]['options'])),'Output':chr(ord('A')+dataset['test'][id]['correct_option'])})
        valid_data=[] 
        for id in range(len(dataset['train'])):valid_data.append({'Input':'Context:{0}\nQuestion:{1}\nOptions:{2}'.format(dataset['train'][id]['context'],dataset['train'][id]['query'],transform_options(dataset['train'][id]['options'])),'Output':chr(ord('A')+dataset['train'][id]['correct_option'])}) #torch.load('{0}_demo.pt'.format(args.task))
        valid_data=valid_data[:valid_num]
        domain='Logic,Commonsense'
    return test_examples,valid_data,domain

logging.basicConfig(
    filename="{0}.log".format(experiment_name), 
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)
#initial_train
def remove_folder(path):
    if os.path.isdir(path):  # Check if the directory exists
        shutil.rmtree(path)
        print(f"Directory '{path}' has been removed.")
    else:
        print(f"Directory '{path}' does not exist.")
test_examples,valid_data,domain=load_task_dataset(task_name,100)  
valid_data=valid_data[:100]
initial_model_path='/dccstor/obsidian_llm/yiduo/copy_v2/finetuned_models/GSM8kbaseline_random_2_20001e-6_model' #GSM8kgsm_baseline_10_20001e-6_model' #'/dccstor/obsidian_llm/yiduo/copy_v2/finetuned_models/GSM8kbaseline_random_2_40001e-6_model'
initial_data_path='dataset_GSM8k_2000baseline_random_2' #dataset_GSM8k_4000baseline_random_2' #'dataset_LogiQA_3000baseline_logiqa'
learning_rate=1e-6
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
for iteration in range(1):
    print('Generating data for iteration ',iteration)
    if iteration==0:
        global_best_model_path=initial_model_path
        global_best_dataset_path=initial_data_path
        global_best_group=[global_best_model_path]
        previous_gradients=None
        try:
            ooa_data=torch.load('ooa_failed_cases_{0}.pt'.format(experiment_name))+torch.load('im_failed_cases_{0}.pt'.format(experiment_name))
        except:
            ooa_failed_cases, im_failed_cases, correct_cases=process_validation_batch_major(global_best_model_path, valid_data,task=task_name,seed=False, iteration=10)  
            torch.save(ooa_failed_cases,'ooa_failed_cases_{0}.pt'.format(experiment_name))
            torch.save(im_failed_cases,'im_failed_cases_{0}.pt'.format(experiment_name))
            ooa_data=torch.load('ooa_failed_cases_{0}.pt'.format(experiment_name))+torch.load('im_failed_cases_{0}.pt'.format(experiment_name))
        global_best_performance=1-len(ooa_data)/(len(valid_data))
        global_best_data=load_from_disk(global_best_dataset_path)
        instructions_data=run_ooa_instruction_optimization(ooa_data,task_instruction,'',task_name,experiment_name+'_'+str(iteration),domain,max_data_num=int(len(global_best_data)*rate),data_num=20)
    else:
        ooa_failed_cases=torch.load('ooa_failed_cases_{0}_{1}.pt'.format(experiment_name,iteration-1))
        im_failed_cases=torch.load('im_failed_cases_{0}_{1}.pt'.format(experiment_name,iteration-1))
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
        global_best_data=load_from_disk(global_best_dataset_path)
        #global_best_data_name=torch.load('global_best_data_name_{0}_{1}.pt'.format(experiment_name,iteration-1))
        instructions_data=run_ooa_instruction_optimization(ooa_data,task_instruction,'',task_name,experiment_name+'_'+str(iteration),domain,max_data_num=int(len(global_best_data)*rate),data_num=20,previous_gradients=previous_gradients)
     #    pdb.set_trace()
    keys=[key for key in instructions_data.keys()]
    #global_best_data=load_from_disk(global_best_dataset_path)
    #global_filter
    def transform_from_low_to_up(data):
            new_data=[]
            for item in data:
                new_data.append({'Input':item['instruction'],'Output':item['output']})
            return new_data
    global_best_data=transform_from_low_to_up(global_best_data)
    '''
    global_data_count=process_validation_batch_count(global_best_model_path, global_best_data,task=task_name,seed=False, iteration=10)
    scores=[item[1] for item in global_data_count]
    median=np.median(scores)
    global_data_count.sort(key=lambda x:abs(x[1]-median))
    global_best_data=[item[0] for item in global_data_count]
    '''
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
 #           from generate_data import majority_voting
#            pdb.set_trace()
            data=[{'input':example['input'][example['input'].find('INPUT')+len('INPUT'):],'output':example['output'].replace("Let's think step by step.\n\n",'').replace('\n\nThe final answer is','####')} for example in examples if example]
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
            data_count=process_validation_batch_count(initial_model_path, data,task=task_name,seed=False, iteration=10)
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
            names_valid[name]=instructions_data[key][1] #global_best_data
            for num in [int((rate/3)*len(global_best_data)),int((rate*2/3)*len(global_best_data)),int(rate*len(global_best_data))]: #            for num in [min(int(len(data)*(1/3)),int(0.03*len(global_best_data))),min(int(len(data)*(2/3)),int(0.05*len(global_best_data))),min(len(data),int(0.1*len(global_best_data)))]:
                cur_data=transform_from_up_to_low(global_best_data)
                cur_data.extend(data[:num])
                #majority_votingcur_data=transform_from_up_to_low(cur_data)
                clean_and_collect_dataset(cur_data,'',experiment_name+name+'_'+str(iteration)+'_'+str(num))
                #key_dataset=load_from_disk(name)
                #dataset_concat=concatenate_datasets([global_best_data,key_dataset])
                #dataset_concat.save_to_disk(experiment_name+name+'_'+str(iteration)+'_'+str(num))
                print(experiment_name+name+'_'+str(iteration)+'_'+str(num),len(cur_data))
                names.append(experiment_name+name+'_'+str(iteration)+'_'+str(num))
                names_dataset[name+'_'+str(iteration)+'_'+str(num)]=experiment_name+name+'_'+str(iteration)+'_'+str(num)
        print("Now train your model on these datasets and collect the validation results,save their names! as {0}_names.pt".format(experiment_name+'_'+str(iteration)))
        torch.save(names,'{0}_names.pt'.format(experiment_name+'_'+str(iteration)))
        torch.save(names_dataset,'{0}_names_dataset.pt'.format(experiment_name+'_'+str(iteration)))
        torch.save(names_valid,'{0}_names_valid.pt'.format(experiment_name+'_'+str(iteration)))
        torch.save(names_keys,'{0}_names_keys.pt'.format(experiment_name+'_'+str(iteration)))
    print('Training models for iteration ',iteration)
    for name in names:
        model_output_path=os.path.join(output_path,name+'_model')
        failed_times=0
        while not train_status_check(model_output_path):
            train_model(os.path.join(cur_path,name),base_model_path,model_output_path,seed=train_seed,learning_rate=learning_rate)
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
        model_paths.extend(global_best_group)
        paths_score=[] #{}
        for path in model_paths:
            f_test,c_test=valid_results_collect(path,test_examples, task_name)
            f_valid,c_valid=valid_results_collect(path,valid_data, task_name)
            paths_score.append((len(c_test)/len(test_examples),len(c_valid)/len(valid_data),str(path))) #,path)
        print("path_score",paths_score)
        win_paths=[path[2] for path in paths_score if path[1]>=paths_score[-1][1]]
        best_path,best_performance=bayesian_search(model_paths,exp_name=experiment_name,base_model_path=base_model_path,task=task_name,valid_data=test_examples)
        f_test,c_test=valid_results_collect(best_path, test_examples, task_name)
        avg_test_acc=len(c_test)/(len(c_test)+len(f_test))
        print(len(c_test)/(len(c_test)+len(f_test)),'avg_acc')
        print("win_paths",win_paths)
        if len(win_paths)>1:
            best_path,best_performance=bayesian_search(win_paths,exp_name=experiment_name,base_model_path=base_model_path,task=task_name,valid_data=test_examples)
            f_test,c_test=valid_results_collect(best_path, test_examples, task_name)
            avg_test_acc=len(c_test)/(len(c_test)+len(f_test))
            print(len(c_test)/(len(c_test)+len(f_test)),'win_avg_acc')
        pdb.set_trace()
        dataset_names=list(set([name.split('_')[0] for name in names_dataset.keys()]))
        model_groups={}
        
        for dataset_name in dataset_names:
            model_group=[model_path for model_path in model_paths if dataset_name in model_path]
            model_group.extend(global_best_group)
            model_groups[dataset_name]=model_group  
        group_best_paths_acc=[]
        for dataset_name in model_groups.keys():
            model_group=model_groups[dataset_name]
            best_path,best_performance=bayesian_search(model_group,exp_name=experiment_name,task=task_name,valid_data=valid_data)
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
        print('cur','global',cur_best_performance,global_best_performance)
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
   
