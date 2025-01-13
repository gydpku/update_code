import torch
from model_eval import process_validation_batch_major,valid_results_collect
import os
import shutil
import subprocess
import argparse
import pdb
import time
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers import LlamaTokenizer
from vllm import LLM, SamplingParams
from datasets import load_dataset
import numpy as np
import optuna
from valid_data_divide import process_nli_validation_batch
def load_models(paths,output_path='/dccstor/obsidian_llm/yiduo/copy_v2/finetuned_models/',base_path='/dccstor/obsidian_llm/yiduo/h100_data/llama-3-8b'):
    model_paths = paths #[os.path.join(output_path, model_name+'_model') for model_name in names]
    
#    model_paths=[os.path.join('/dccstor/obsidian_llm/yiduo/copy_v2/finetuned_models/random_models_9','randoms_data_3k_model_{0}'.format(number)) for number in range(2020,2029)]
    models=[]
    for model_path in model_paths:
        models.append(AutoModelForCausalLM.from_pretrained(model_path))
#    pdb.set_trace()
    base_model = AutoModelForCausalLM.from_pretrained(base_path)
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    return models,base_model,tokenizer

def assign_weights(models,weights):

    weight_state_dict = {}
    for key in models[0].state_dict().keys():
        weight_state_dict[key] = sum([model.state_dict()[key]*weight for model,weight in zip(models,weights)])
    return weight_state_dict
def remove_folder(path):
    if os.path.isdir(path):  # Check if the directory exists
        shutil.rmtree(path)
        print(f"Directory '{path}' has been removed.")
    else:
        print(f"Directory '{path}' does not exist.")
def save_as_model(base_model,tokenizer,weight_state_dict,weights,exp_name,output_path='/dccstor/obsidian_llm/yiduo/copy_v2/finetuned_models/'):
    base_model.load_state_dict(weight_state_dict)
    weight_strs = [f"{w:.4f}" for w in weights]
    model_path=f"weights_{exp_name}_{''.join(weight_strs)}"
    model_path=os.path.join(output_path,model_path)
    base_model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    return model_path
def bayesian_search(names,exp_name,base_model_path,task='nli',valid_data=None,output_path='/dccstor/obsidian_llm/yiduo/copy_v2/finetuned_models/',m_voting_num=1):
    # Run Optuna optimization
    start_time=time.time()
    '''
    if 'nli' in task.lower():
        valid_data_cot=torch.load('nli_demo.pt')
        test_examples=[]
        dataset=load_dataset('hippocrates/MedNLI_test')
#    pdb.set_trace()
        for id in range(len(dataset['test'])):
            test_examples.append({'Input':dataset['test'][id]['query'],'Output':dataset['test'][id]['answer']})
        valid_data=valid_data_cot #test_examples
    '''
    models,base_model,tokenizer=load_models(names,base_path=base_model_path)
    '''
#    names.append('no_task_2')
    model_paths = [os.path.join(output_path, model_name+'_model') for model_name in names]
 #   pdb.set_trace() #model_paths.append('no_task_2')
    names_valid_cases={}
    for name in names_valid:
        valid_cases=names_valid[name]
        ooa_failed_cases, im_failed_cases, correct_cases=process_nli_validation_batch(model_paths[-1], valid_cases,seed=False, iteration=1)
        names_valid_cases[name]=(valid_cases,len(correct_cases),ooa_failed_cases+im_failed_cases)
  #  pdb.set_trace()
    _,_,valid_cot_correct=process_nli_validation_batch(model_paths[-1], valid_data_cot,seed=False, iteration=1)
    adv_models=[]
    prev_best_num=len(valid_cot_correct)
    adv_models.append(models[-1])
    records={}    #pdb.set_trace()
    valid_names=[name for name in names_valid_cases]
    for model_id,model in enumerate(models[:-1]):
                name=valid_names[model_id//(len(models[:-1])//len(valid_names))]
                ooa_failed_cases, _, correct_cases_new=process_nli_validation_batch(model_paths[model_id], names_valid_cases[name][0],seed=False, iteration=1)
                _,_,valid_cot_correct=process_nli_validation_batch(model_paths[model_id], valid_data_cot,seed=False, iteration=1)
                records[model_paths[model_id]]=(len(correct_cases_new),names_valid_cases[name][1],len(valid_cot_correct),prev_best_num) #                pdb.set_trace()
                if len(correct_cases_new)>names_valid_cases[name][1] and len(valid_cot_correct)>=prev_best_num:
                    adv_models.append(model)
                elif len(valid_cot_correct)>prev_best_num:
                    adv_models.append(model)
    models=adv_models    
    print(records)
    print(len(models))
  
    pdb.set_trace()
    '''
    end_time=time.time()
    models_num=len(models) #    pdb.set_trace()
    print("Loading model",end_time-start_time)
    def objective(trial):
    # Suggest weights for the models
        weights=[]  
        for i in range(models_num):
            weights.append(trial.suggest_uniform(f"w{i}", 0, 1))
        
        # Normalize weights to sum to 1
        weights = np.array(weights)
        weights = weights / weights.sum()
        print('weights',weights)
        def evaluate_performance(weights,valid_data):
            start_time=time.time()
            weight_state_dict=assign_weights(models,weights)
            end_time=time.time()
            print("Weighting model",end_time-start_time)
            start_time=time.time()
            model_path=save_as_model(base_model,tokenizer,weight_state_dict,weights,exp_name)
            end_time=time.time()
            print('Saving model',end_time-start_time)
            start_time=time.time()
#            pdb.set_trace()
            f_test,c_test=valid_results_collect(model_path,valid_data, task)
            #_,_,c_test=process_validation_batch_major(model_path, valid_data, task,seed=False,iteration=m_voting_num,temperature=0.0) #valid_results_collect(model_path, valid_data, task)
            end_time=time.time()
            print('Testing time',end_time-start_time)
            remove_folder(model_path)
            print('weights',weights,'Performance',len(c_test)/len(valid_data))
            return len(c_test)/len(valid_data)
        performance=evaluate_performance(weights,valid_data)
        return performance
    study = optuna.create_study(direction="maximize")  # Change direction to 'maximize' for accuracy
    study.optimize(objective, n_trials=100)
    best_weights = study.best_params
    best_weights_list=[best_weights[key] for key in best_weights.keys()] #best_weights_list=[value for key, value in best_weights.item()]
    weights = np.array(best_weights_list)
    weights = weights / weights.sum()
    best_weights_list=list(weights)
    weight_state_dict=assign_weights(models,best_weights_list)
    best_model_path=save_as_model(base_model,tokenizer,weight_state_dict,best_weights_list,exp_name)
    print("Best Weights:", best_weights)
    print("Best Performance (Accuracy):", study.best_value)
    return best_model_path,study.best_value





