

from ooa_instruction_optimization_constrast_t_3 import run_ooa_instruction_optimization #,run_ooa_instruction_optimization_loop
task_instruction="The task is to generate medical inference data based on the provided medical passage."
from valid_data_analysis import data_type_analysis
from datasets import load_dataset
from model_soups_method import find_best_combination
from bayesian_soup_search import bayesian_search
import torch
from openai_call import query_azure_openai_chatgpt_chat
import pdb
from datasets import concatenate_datasets
from datasets import load_from_disk
from generate_data import clean_and_collect_dataset,data_sample_pattern
import logging
import os
import shutil
from model_eval import process_validation_batch_major,valid_results_collect
from auto_finetune import train_model,train_status_check
print('Starting')
experiment_name='baseline_logiqa'
output_path='/dccstor/obsidian_llm/yiduo/copy_v2/finetuned_models/'
cur_path='/dccstor/obsidian_llm/yiduo/summary/src/'
model='LLama3-8B'
base_model_path='/dccstor/obsidian_llm/yiduo/llama-3-instruct' #'/dccstor/obsidian_llm/yiduo/h100_data/llama-3-8b'
task_name='LogiQA'
train_seed=2024
task_instruction='You should answer logical reasoning questions accurately based on the provided context.' #'You are given a word problem involving basic arithmetic, algebra, or geometry. Your task is to carefully read the problem and provide a step-by-step solution, ensuring that all intermediate steps are shown clearly.'
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
valid_acc=[0]
best_valid_acc=0
best_model_path=''
best_data_path=''
print('start finding')

for data_num in [2000,3000,4000,5000,6000,7000,8000,9000,10000]:
    store_name='Synthetic_initial_'+task_name+experiment_name
    print('sampling: ',data_num)
    data=data_sample_pattern(task_instruction,domain,data_num,store_name,'',demo_examples=valid_data,temperature=0.7,task_name=task_name,neg_sample=True,pattern=False,voting=True,iteration_number=1,sample_demo_num=3,passage_num=10000,valid_num=100,types=None)
    data=[example for example in data if example]
    dataset_name='dataset_'+task_name+'_'+str(data_num)+experiment_name
    clean_and_collect_dataset(data,'',dataset_name)
    cur_output_path=os.path.join(output_path,task_name+experiment_name+'_'+str(data_num)+'1e-6_model') #name+'_model')
    failed_times=0
    while not train_status_check(cur_output_path):
        train_model(os.path.join(cur_path,dataset_name),base_model_path,cur_output_path,seed=train_seed,learning_rate=1e-6)
        if not train_status_check(cur_output_path):
            remove_folder(cur_output_path)   
            failed_times+=1
            print('We have failed',failed_times,'times')
            if failed_times>3:
                pdb.set_trace()
    print('We have finished training the model ',cur_output_path)
    ooa_failed_cases, im_failed_cases, correct_cases=process_validation_batch_major(cur_output_path, valid_data,task=task_name,seed=False, iteration=100)
    valid_acc.append(len(correct_cases)/(len(correct_cases)+len(ooa_failed_cases)+len(im_failed_cases)))
    f_test,c_test=valid_results_collect(cur_output_path, test_examples, task_name)
    if valid_acc[-1]>best_valid_acc:
        best_valid_acc=valid_acc[-1]
        best_model_path=cur_output_path
        best_data_path=os.path.join(cur_path,dataset_name)
    print('cur_acc is ',valid_acc[-1],'test_acc is', len(c_test)/len(test_examples))
    torch.save(ooa_failed_cases,'ooa_failed_cases_{0}.pt'.format(experiment_name))
    torch.save(im_failed_cases,'im_failed_cases_{0}.pt'.format(experiment_name))

initial_model_path=best_model_path
initial_data_path=best_data_path
print('The best valid accuracy is',best_valid_acc,'at',data_num,'data',initial_model_path,initial_data_path)
pdb.set_trace()
print(initial_model_path,initial_data_path) 
# Configure logging

def remove_previous_folders(experiment_name,iteration):
    if iteration>0:
        prev_names=torch.load('{0}_names.pt'.format(experiment_name+'_'+str(iteration-1)))
        prev_model_paths=[os.path.join(output_path, model_name+'_model') for model_name in prev_names]
        global_best_path=torch.load('global_best_model_path_{0}_{1}.pt'.format(experiment_name,iteration-1))
        prev_best_path=torch.load('cur_best_path_{0}_{1}.pt'.format(experiment_name,iteration-1))
        for path in prev_model_paths:
            if path!=prev_best_path and path!=global_best_path:
                remove_folder(path)

for iteration in range(5):
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
    global_best_data=load_from_disk(initial_data_path)
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
            print(len(examples),len(data))
            prompt="make a name (one word) for this characteristic, directly and only output the name:{0}".format(key)
            name=query_azure_openai_chatgpt_chat(prompt)
            
            names_valid[name]=instructions_data[key][1]
            for num in [100,300,500]:
                clean_and_collect_dataset(data[:num],'',name)
                key_dataset=load_from_disk(name)
                dataset_concat=concatenate_datasets([global_best_data,key_dataset])
                dataset_concat.save_to_disk(experiment_name.replace('random_3k_','')+name+'_'+str(iteration)+'_'+str(num))
                print(experiment_name.replace('random_3k_','')+name+'_'+str(iteration)+'_'+str(num))
                names.append(experiment_name.replace('random_3k_','')+name+'_'+str(iteration)+'_'+str(num))
                names_dataset[name+'_'+str(iteration)+'_'+str(num)]=experiment_name.replace('random_3k_','')+name+'_'+str(iteration)+'_'+str(num)
        print("Now train your model on these datasets and collect the validation results,save their names! as {0}_names.pt".format(experiment_name+'_'+str(iteration)))
        torch.save(names,'{0}_names.pt'.format(experiment_name+'_'+str(iteration)))
        torch.save(names_dataset,'{0}_names_dataset.pt'.format(experiment_name+'_'+str(iteration)))
        torch.save(names_valid,'{0}_names_valid.pt'.format(experiment_name+'_'+str(iteration)))
    #training models
    for name in names:
        import time
        output_path='/dccstor/obsidian_llm/yiduo/copy_v2/finetuned_models/'
        cur_path='/dccstor/obsidian_llm/yiduo/summary/src/'
        base_model_path='/dccstor/obsidian_llm/yiduo/h100_data/llama-3-8b'
        model_output_path=os.path.join(output_path,name+'_model')
        train_model(os.path.join(cur_path,name),base_model_path,model_output_path,seed=train_seed)
        failed_times=0
        while not train_status_check(model_output_path):
            
            train_model(os.path.join(cur_path,name),base_model_path,model_output_path,seed=train_seed)
            if not train_status_check(model_output_path):
                remove_folder(model_output_path)
                failed_times+=1
                print('We have failed',failed_times,'times')
            if failed_times>3:
                pdb.set_trace()
        print('We have finished training the model ',model_output_path)
    try:
        global_best_model_path=torch.load('global_best_model_path_{0}_{1}.pt'.format(experiment_name,iteration))
        cur_best_path=torch.load('cur_best_path_{0}_{1}.pt'.format(experiment_name,iteration))
    except:
        #global_best_model_path
        model_paths = [os.path.join(output_path, model_name+'_model') for model_name in names]
        model_paths.append(global_best_model_path)
        best_path,best_performance=bayesian_search(model_paths,exp_name=experiment_name,task='nli') #,exp_name=experiment_name)
        global_best_model_path=best_path
        f_test,c_test=valid_results_collect(global_best_model_path, test_examples, task_name)
        avg_test_acc=len(c_test)/(len(c_test)+len(f_test))
        print(len(c_test)/(len(c_test)+len(f_test)),'avg_acc')
        torch.save(global_best_model_path,'global_best_model_path_{0}_{1}.pt'.format(experiment_name,iteration))
        torch.save(global_best_model_path,'cur_best_path_{0}_{1}.pt'.format(experiment_name,iteration))
        logging.info("Logging important variables:")
        logging.info(f"Iteration {iteration}: global_best_model_path = {global_best_model_path}")
        logging.info(f"Iteration {iteration}: names = {names}")
        logging.info(f"Iteration {iteration}: names_dataset = {names_dataset}")
        logging.info(f"Iteration {iteration}: avg_test_acc = {avg_test_acc}")
        logging.info(f"Iteration {iteration}: avg_valid_acc = {best_performance}")
    #search_name=experiment_name+'_'+str(iteration)+'all_mix_test'
    #best_path=bayesian_search(names,names_valid,task='nli') #find_best_combination(model_path,test_examples,test_examples, search_name,seed=False,task='nli')
    #pdb.set_trace() #best_path='/dccstor/obsidian_llm/yiduo/copy_v2/finetuned_models/no_task_2_model' #print(best_path) #   best_path='/dccstor/obsidian_llm/yiduo/copy_v2/finetuned_models/weights_0.04260.18050.00600.08370.12520.10700.01660.02640.19610.2160' #'/dccstor/obsidian_llm/yiduo/summary/src/weights_0.17930.75890.02520.35220.52630.44980.06980.11100.82460.9085'
    ooa_failed_cases, im_failed_cases, correct_cases=process_validation_batch_major(global_best_model_path, valid_data,task=task_name,seed=False, iteration=100)
    #pdb.set_trace()
    torch.save(ooa_failed_cases,'ooa_failed_cases_{0}_{1}.pt'.format(experiment_name,iteration))
    torch.save(im_failed_cases,'im_failed_cases_{0}_{1}.pt'.format(experiment_name,iteration))
#    pdb.set_trace()
