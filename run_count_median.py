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
from openai_call import query_azure_openai_chatgpt_chat
import pdb
from datasets import concatenate_datasets
from datasets import load_from_disk
from generate_data import clean_and_collect_dataset
from model_eval import process_validation_batch_major,valid_results_collect,process_validation_batch_count
import logging
import os
import shutil
experiment_name='random_3k_baseline_count_median'
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
for iteration in range(5):
    print('Generating data for iteration ',iteration)
    if iteration==0:
        global_best_model_path=initial_model_path
        global_best_data_name=initial_data_path
        previous_gradients=None
        try:
            ooa_data=torch.load('ooa_failed_cases_{0}.pt'.format(experiment_name))+torch.load('im_failed_cases_{0}.pt'.format(experiment_name))
        except:
            ooa_failed_cases, im_failed_cases, correct_cases=process_validation_batch_major(global_best_model_path, valid_data,task=task_name,seed=False, iteration=100)  
    #pdb.set_trace()
            torch.save(ooa_failed_cases,'ooa_failed_cases_{0}.pt'.format(experiment_name))
            torch.save(im_failed_cases,'im_failed_cases_{0}.pt'.format(experiment_name))
            ooa_data=torch.load('ooa_failed_cases_{0}.pt'.format(experiment_name))+torch.load('im_failed_cases_{0}.pt'.format(experiment_name))
        instructions_data=run_ooa_instruction_optimization(ooa_data,task_instruction,'','nli',experiment_name+'_'+str(iteration),'Medical',data_num=20)
    else:
        ooa_data=torch.load('ooa_failed_cases_{0}_{1}.pt'.format(experiment_name,iteration-1))+torch.load('im_failed_cases_{0}_{1}.pt'.format(experiment_name,iteration-1))
        previous_gradients=[]
        for i in range(iteration):
            previous_char=torch.load(experiment_name+"_"+str(i)+"_characteristic_dict.pt")
            previous_gradients.extend([key for key in previous_char.keys()])
#        pdb.set_trace()
        global_best_model_path=torch.load('global_best_model_path_{0}_{1}.pt'.format(experiment_name,iteration-1))


        #global_best_data_name=torch.load('global_best_data_name_{0}_{1}.pt'.format(experiment_name,iteration-1))
        instructions_data=run_ooa_instruction_optimization(ooa_data,task_instruction,'','nli',experiment_name+'_'+str(iteration),'Medical',data_num=20,previous_gradients=previous_gradients)
     #    pdb.set_trace()
    keys=[key for key in instructions_data.keys()]
    global_best_data=load_from_disk(initial_data_path) #global_best_data_name)
    print('Constructing training data for iteration ',iteration)
    try:
        names=torch.load('{0}_names.pt'.format(experiment_name+'_'+str(iteration)))
        names_valid=torch.load('{0}_names_valid.pt'.format(experiment_name+'_'+str(iteration)))
        names_dataset=torch.load('{0}_names_dataset.pt'.format(experiment_name+'_'+str(iteration)))
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
            data_count=process_validation_batch_count(global_best_model_path, data,task=task_name,seed=False, iteration=100)
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
            names_valid[name]=instructions_data[key][1]
            for num in [100,300,500]:
                clean_and_collect_dataset(data[:num],'',name)
                key_dataset=load_from_disk(name)
                dataset_concat=concatenate_datasets([global_best_data,key_dataset])
                dataset_concat.save_to_disk(experiment_name+name+'_'+str(iteration)+'_'+str(num))
                print(experiment_name+name+'_'+str(iteration)+'_'+str(num))
                names.append(experiment_name.replace('random_3k_','')+name+'_'+str(iteration)+'_'+str(num))
                names_dataset[name+'_'+str(iteration)+'_'+str(num)]=experiment_name+name+'_'+str(iteration)+'_'+str(num)
        print("Now train your model on these datasets and collect the validation results,save their names! as {0}_names.pt".format(experiment_name+'_'+str(iteration)))
        torch.save(names,'{0}_names.pt'.format(experiment_name+'_'+str(iteration)))
        torch.save(names_dataset,'{0}_names_dataset.pt'.format(experiment_name+'_'+str(iteration)))
        torch.save(names_valid,'{0}_names_valid.pt'.format(experiment_name+'_'+str(iteration)))
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
        model_paths.append(global_best_model_path)
        ''' 
        model_cases_list = []
        
        for model_path in model_paths:
            ooa_failed_cases, im_failed_cases, correct_cases = process_validation_batch_major(model_path, 
                                                                                              valid_data,task=task_name,
                                                                                              seed=False, iteration=100)
            
            c_cases=[(case[0]['Input'],case[0]['Output']) for case in correct_cases]
            model_cases_list.append((model_path, c_cases))
            #model_cases_list.append((model_path, correct_cases))  # Fixed tuple append syntax
        torch.save(model_cases_list,'model_cases_list_{0}_{1}.pt'.format(experiment_name,iteration))
        iteration_num=iteration-1
        while iteration_num>=0:
            model_cases_list.extend(torch.load('model_cases_list_{0}_{1}.pt'.format(experiment_name,iteration_num)))
            iteration_num-=1
        
        # Sort models by number of correct cases
        model_cases_list.sort(key=lambda x: len(x[1]), reverse=True)
        
        # Track verified models and their collective performance
        verified_model_paths = []
        global_correct_set = set()
        
        # Evaluate each model's contribution to overall coverage
        for model_path, correct_cases in model_cases_list:
            current_cases = set(correct_cases)
            combined_coverage = global_correct_set | current_cases
            
            # Add model if it improves overall coverage
            if len(combined_coverage) > len(global_correct_set):
                verified_model_paths.append(model_path)
                global_correct_set = combined_coverage
            else:
                remove_folder(model_path)  #        pdb.set_trace()
        best_path,best_performance=bayesian_search(verified_model_paths,task='nli')
        '''

        best_path,best_performance=bayesian_search(model_paths,task='nli',m_voting_num=10)
        global_best_model_path=best_path
        f_test,c_test=valid_results_collect(global_best_model_path, test_examples, task_name)
        avg_test_acc=len(c_test)/(len(c_test)+len(f_test))
        print(len(c_test)/(len(c_test)+len(f_test)),'avg_acc')
        torch.save(global_best_model_path,'global_best_model_path_{0}_{1}.pt'.format(experiment_name,iteration))
        torch.save(global_best_model_path,'cur_best_path_{0}_{1}.pt'.format(experiment_name,iteration))
        logging.info("Logging important variables:")
        logging.info(f"Iteration {iteration}: global_best_model_path = {global_best_model_path}")
#        logging.info(f"Iteration {iteration}: global_best_data_name = {global_best_data_name}")
        logging.info(f"Iteration {iteration}: names = {names}")
        logging.info(f"Iteration {iteration}: names_dataset = {names_dataset}")
        logging.info(f"Iteration {iteration}: best_performance = {best_performance}")
        logging.info(f"Iteration {iteration}: avg_test_acc = {avg_test_acc}")
    #search_name=experiment_name+'_'+str(iteration)+'all_mix_test'
    #best_path=bayesian_search(names,names_valid,task='nli') #find_best_combination(model_path,test_examples,test_examples, search_name,seed=False,task='nli')
    #pdb.set_trace() #best_path='/dccstor/obsidian_llm/yiduo/copy_v2/finetuned_models/no_task_2_model' #print(best_path) #   best_path='/dccstor/obsidian_llm/yiduo/copy_v2/finetuned_models/weights_0.04260.18050.00600.08370.12520.10700.01660.02640.19610.2160' #'/dccstor/obsidian_llm/yiduo/summary/src/weights_0.17930.75890.02520.35220.52630.44980.06980.11100.82460.9085'
    ooa_failed_cases, im_failed_cases, correct_cases=process_validation_batch_major(global_best_model_path, valid_data,task=task_name,seed=False, iteration=100)  
    #process_nli_validation_batch_major(global_best_model_path, valid_data,seed=False, iteration=100)
    #pdb.set_trace()
    torch.save(ooa_failed_cases,'ooa_failed_cases_{0}_{1}.pt'.format(experiment_name,iteration))
    torch.save(im_failed_cases,'im_failed_cases_{0}_{1}.pt'.format(experiment_name,iteration))
#    pdb.set_trace()
