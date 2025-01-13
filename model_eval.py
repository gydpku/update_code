from vllm import LLM, SamplingParams
import multiprocessing
import time
import gc
import torch
import re
import pdb
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import gc
import pdb
import torch
import time
from collections import Counter
from vllm import LLM, SamplingParams
#from openai_call import query_azure_openai_chatgpt_chat
def label_transform(label):
    if label==1:
        return 'neutral'
    if label==0:
        return 'entailment'
    if label==2:
        return 'contradiction'
sampling_params = SamplingParams(temperature=0.0,max_tokens=600, top_p=0.95)
def valid_results_collect(model_path,valid_data,task):
   
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
#    multiprocessing.set_start_method('spawn')
    trained_model=LLM(model=model_path,gpu_memory_utilization=0.95)
    
    start_t=time.time()
    if task=='sql':
        failed_cases,correct_cases=sql_evaluation(trained_model,valid_data)
    elif 'logiqa' in task.lower() or 'nli' in task.lower() or 'gsm8k' in task.lower():
        failed_cases,correct_cases=batched_evaluation(trained_model,valid_data,task)
    del trained_model
    end_t=time.time()
    print('time',end_t-start_t)
    gc.collect()  # Run garbage collection to free up memory
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.synchronize()
    #torch.cuda.synchronize()
    #torch.cuda.empty_cache()
    #torch.cuda.synchronize()
    time.sleep(10)
    return failed_cases,correct_cases
def extract_answer_prediction_nli(predicted_output):
    sens=predicted_output.split('.')
    final_sens=[sen for sen in sens if 'final' in sen]
    for sen in final_sens:
        if extract_answer_nli(sen):
            return extract_answer_nli(sen)
    return
def extract_answer_prediction_logiqa(predicted_output):
    sens=predicted_output.split('.')
    final_sens=[sen for sen in sens if 'final' in sen]
    for sen in final_sens:
        if extract_answer_logiqa(sen):
    #        print('extract',sen,'\n',extract_answer_logiqa(sen))
            return extract_answer_logiqa(sen)
    return
def extract_answer_prediction_gsm8k(predicted_output):
    #print('predicted_output',predicted_output,'\n')
    if '####' in predicted_output:
        predicted_output=predicted_output[predicted_output.find('####'):]
    elif '###' in predicted_output:
        predicted_output=predicted_output[predicted_output.find('###'):]
    elif '##' in predicted_output:
        predicted_output=predicted_output[predicted_output.find('##'):]
    #else:
        #predicted_output=predicted_output.split('\n\n')[-2]
        
    regex_pattern = "(-?[$0-9.,]{2,})|(-?[0-9]+)"
    regexes_to_ignore =[
        ",",
        "\\$",
        "(?s).*#### ",
        "\\.$"
    ]
    match = re.findall(regex_pattern, predicted_output)
    if match:
        print(predicted_output,'\n')
        print(match,'\n')
        print(match[-1],'\n')
        match = match[-1]
        if isinstance(match, tuple):
            match = [m for m in match if m][0]
        text = match.strip()

        for regex in regexes_to_ignore:
            text = re.sub(regex, "", text)
#        print(text,predicted_output)
        if text.count('.') > 1:
        # Retain only up to the last valid dot
            index_1=text.find('.')
            index_2=text.find('.',index_1+1)
            text = text[:index_2]
        try:
            return float(text)
        except ValueError as e:
            print('Error:', e)
            return None
    else:
        return None
def extract_answer_nli(text):
    if 'neutral' in text.lower():
        return 'neutral'
    if 'contradiction' in text.lower():
        return 'contradiction'
    if 'entailment' in text.lower():
        return 'entailment'
    return None
def extract_answer_logiqa(text):
    matches = re.findall(r'\b[A-D]\b', text)
    if matches:
        return matches[0]
    return None

def extract_answer_gsm8k(text):
    try_1=extract_answer_prediction_gsm8k(text)
    
    if try_1 is not None:
        return try_1
    else:
        if len(text.split('\n\n'))>1:
          return extract_answer_prediction_gsm8k(text.split('\n\n')[-2])
        else:
          return
    return try_1 if try_1 is not None else extract_answer_prediction_gsm8k(text.split('\n\n')[-2])
    #return extract_answer_prediction_gsm8k(text) #float(text.split('####')[-1].strip().replace(',',''))
def process_batch(data_batch,task,trained_model,failed_cases,correct_cases):
    batch_prompts = [data['Input'] for data in data_batch]
    if 'gsm8k' in task.lower():
        #demo_prompt='Here are some demonstration examples:'
        #demos=torch.load('GSM8k_demo.pt')
        #for id in range(8):
        #    demo=demos[id]
        #    demo_prompt+='\n Example {2}: Input:{0} Output:{1}'.format(demo['Input'],demo['Output'],id+1)
        batch_prompts = [data['Input']+"\n Let's think step by step. At the end, you MUST write the answer as an number after '####' likes '#### number'." for data in data_batch]
    if 'nli' in task.lower():
        aaa=torch.load('MedNLI_demo.pt')
        batch_prompts = [data['Input']+'Here are some examples:'+'Example 0:'+str(aaa[0])+'\nExample 1:'+str(aaa[1])+'\nExample 2:'+str(aaa[2])+'\nExample 3:'+str(aaa[3])+'\nExample 4:'+str(aaa[4])+"\n End your response with 'The final answer is xxx'." for data in data_batch] 
    outputs = trained_model.generate(batch_prompts, sampling_params)
    if 'nli' in task.lower():
        labels=['entailment','contradiction','neutral']
        for data, output in zip(data_batch, outputs):

            predicted_output = output.outputs[0].text
            predicted_res = extract_answer_prediction_nli(predicted_output)
            label = extract_answer_nli(data['Output'].split('is')[-1])
            if not predicted_res:
                predicted_res=predicted_output
            non_labels = [lbl for lbl in labels if lbl != label]
         #   print('matched',label,predicted_res,'\n')
            if label not in predicted_res or any(non_label in predicted_res for non_label in non_labels):
                failed_cases.append((data['Input'],predicted_res,label,data))
            else:
                correct_cases.append((data['Input'],predicted_res,label,data))
    elif 'gsm8k' in task.lower():
        for data, output in zip(data_batch, outputs):
            predicted_output = output.outputs[0].text
            predicted_res = extract_answer_prediction_gsm8k(predicted_output)
            label = extract_answer_gsm8k(data['Output'])
            print(predicted_output,'\n')
            print('pred extraction and label',predicted_res,label,'\n')
#            pdb.set_trace()
#            if label is None:
            if predicted_res is None or label is None:
                failed_cases.append((data['Input'],predicted_res,label,data))
            
            elif abs(predicted_res-label)>1e-3:
                failed_cases.append((data['Input'],predicted_res,label,data))
            else:
                correct_cases.append((data['Input'],predicted_res,label,data))
    elif 'logiqa' in task.lower():
        for data, output in zip(data_batch, outputs):
            predicted_output = output.outputs[0].text
            predicted_res = extract_answer_prediction_logiqa(predicted_output)
            label = extract_answer_logiqa(data['Output'])
            print(predicted_res,label,'\n')
            print('pred and result',predicted_res,label,'\n')
            if predicted_res is None:
                failed_cases.append((data['Input'],predicted_res,label,data))
            elif predicted_res != label:
                failed_cases.append((data['Input'],predicted_res,label,data))
            else:
                correct_cases.append((data['Input'],predicted_res,label,data))
    return failed_cases,correct_cases
def batched_evaluation(trained_model,valid_data,task):
    id=0
    failed_cases=[]
    correct_cases=[]
    batch_size=500
    batched_data = [valid_data[i:i+batch_size] for i in range(0, len(valid_data), batch_size)]
    for batch in batched_data:
        failed_cases,correct_cases=process_batch(batch,task,trained_model,failed_cases,correct_cases)
    return failed_cases,correct_cases

def sql_evaluation(trained_model,valid_data):
    id=0
    failed_cases=[]
    correct_cases=[]
    for triple in valid_data:
        
        db_id,prompt,ground_truth=triple
        prompt=prompt.replace('SELECT','')
        db_path='/dccstor/obsidian_llm/yiduo/AgentBench/DAMO-ConvAI/bird/data/train/train_databases/{0}/{0}.sqlite'.format(db_id)
        prompt+=' To generate the SQL query to' #print(db_path) #pdb.set_trace()
        conn = sqlite3.connect(db_path)
        output=trained_model.generate(prompt, sampling_params) #pdb.set_trace()
        predicted_sql = output[0].outputs[0].text
        #pdb.set_trace()
        prior_pred=predicted_sql.split('final SQL')[0]
        try:
            predicted_sql = predicted_sql.split('final SQL')[1].strip()
        except:
            predicted_sql = 'SELECT'+predicted_sql.split('SELECT')[1]
        predicted_sql=predicted_sql.split(';')[0]
        predicted_sql=predicted_sql[predicted_sql.find('SELECT'):] #[1:]
        cursor = conn.cursor()
    #    pdb.set_trace()
        try:
            cursor.execute(predicted_sql)
            predicted_res = cursor.fetchall()
            cursor.execute(ground_truth)
            ground_truth_res = cursor.fetchall()
    #print('results',predicted_res,'truth',ground_truth_res,'\n')
            if set(predicted_res) != set(ground_truth_res):
                failed_cases.append((id,prompt,prior_pred+predicted_sql,valid_data[id],ground_truth,predicted_res,ground_truth_res))
            else:
                correct_cases.append((id,prompt,prior_pred+predicted_sql,valid_data[id],ground_truth,predicted_res,ground_truth_res))
        except Exception as e:
            failed_cases.append((id,prompt,predicted_sql,valid_data[id],ground_truth,str(Exception)+str(e)))
        return failed_cases,correct_cases
    
def load_and_generate_predictions(model_path, batch_prompts,seed=True, seeds=['2023', '2024', '2025'], iteration=5,temperature=0.2):
    """Handle model loading and prediction generation"""
    sampling_params = SamplingParams(temperature=temperature, max_tokens=600, top_p=0.95)
    outputs = []
    if seed is False:
        trained_model = LLM(model=model_path, gpu_memory_utilization=0.95)
        for _ in range(iteration):
            output = trained_model.generate(batch_prompts, sampling_params)
            outputs.append(output)
        del trained_model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
        return outputs
    for seed in seeds:
        model_path_seed = f"{model_path}_{seed}"
        trained_model = LLM(model=model_path_seed, gpu_memory_utilization=0.95)
        
        try:
            for _ in range(iteration):
                output = trained_model.generate(batch_prompts, sampling_params)
                outputs.append(output)
        finally:
            # Cleanup resources
            del trained_model
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
            time.sleep(10)
    
    return outputs
def safty_float_check(pred,label):
    if pred is None:
        return True
    if label is None:
        return False
    if abs(float(pred)-float(label))>1e-3:
        return True
    return False
def process_single_example(data, outputs, id, task):
    """Process predictions for a single example"""
    if 'nli' in task.lower():
        label = extract_answer_nli(data['Output'].split('is')[-1])
    elif 'gsm8k' in task.lower():
        label = extract_answer_gsm8k(data['Output'])
    elif 'logiqa' in task.lower():
        label = extract_answer_logiqa(data['Output']) #data['Output']
    predictions = []
    predicted_outputs = []
    #print('labels',data['Output'],'\n',extract_answer_prediction_logiqa(data['Output']),'\n',extract_answer_logiqa(data['Output']))
    # Collect predictions
    for output in outputs:
#        pdb.set_trace()
        predicted_output = output[id].outputs[0].text
        if 'nli' in task.lower():
            predicted_res = extract_answer_prediction_nli(predicted_output)
            if not predicted_res:
                predicted_res = extract_answer_nli(predicted_output)
        elif 'gsm8k' in task.lower():
            predicted_res = extract_answer_prediction_gsm8k(predicted_output)
        elif 'logiqa' in task.lower():
            predicted_res = extract_answer_logiqa(predicted_output)
        predictions.append(predicted_res)
        predicted_outputs.append(predicted_output)
    #print('preds',predictions,'\n','answer',label,'\n')
    # Get incorrect predictions
    if 'nli' in task.lower():
        incorrect_outputs = [
        output
            for output, pred in zip(predicted_outputs, predictions)
                if pred != label
            ]
        i_o=[]
        for output in incorrect_outputs:
            if "Let's think step by step." in output:
                i_o.append(output.split("Let's think step by step.")[1])
            else:
                i_o.append(output)
    elif 'logiqa' in task.lower():
        incorrect_outputs= [
            output
            for output, pred in zip(predicted_outputs, predictions)
                if pred != label
        ]
        i_o=[]

        for output in incorrect_outputs:
            if "Let's think step by step." in output:
                i_o.append(output.split("Let's think step by step.")[1])
            else:
                i_o.append(output)
    elif 'gsm8k' in task.lower():
        incorrect_outputs= [
            output
            for output, pred in zip(predicted_outputs, predictions)
                if safty_float_check(pred,label)
        ]
        
        i_o=[]
        for output in incorrect_outputs:
            if "Let's think step by step." in output:
                i_o.append(output.split("Let's think step by step.")[1])
            else:
                i_o.append(output)
  
    prediction_counts = Counter(predictions)
    majority_prediction = prediction_counts.most_common(1)[0][0]
#    pdb.set_trace()    
    return label, majority_prediction, i_o, predicted_outputs,predictions

def process_validation_batch_major(model_path, data_batch,task,seed=True,  iteration=5,temperature=0.2):
    """Main function to process NLI validation batch"""
    batch_prompts = [data['Input'] for data in data_batch]
    if 'gsm8k' in task.lower():
        batch_prompts = [data['Input']+"\n Let's think step by step. At the end, you MUST write the answer as an number after '####' likes '#### number'." for data in data_batch]
    outputs = load_and_generate_predictions(model_path, batch_prompts,seed=seed, iteration=iteration,temperature=temperature)
    ooa_failed_cases = []  # out-of-agreement failures
    im_failed_cases = []   # inconsistent-majority failures
    correct_cases = []

    for id, data in enumerate(data_batch):
        label, majority_prediction, incorrect_outputs, predicted_outputs,predictions = process_single_example(
            data, outputs, id,task
        )
        print(majority_prediction,label,'\n')
#        pdb.set_trace()
        # Categorize results
        if 'nli' in task.lower():
            if majority_prediction != label:
                # Out of agreement failure - majority prediction is wrong
    #           pdb.set_trace()
                ooa_failed_cases.append((data, incorrect_outputs))

            else:
                # All predictions correct
                correct_cases.append((data, predicted_outputs))
        elif 'gsm8k' in task.lower():
#            print(majority_prediction,label,'\n')
 #           pdb.set_trace()
            if majority_prediction is not None:
                if not safty_float_check(majority_prediction,label):
                    correct_cases.append((data, predicted_outputs))
                else:
                    ooa_failed_cases.append((data, incorrect_outputs))
            else:
                ooa_failed_cases.append((data, incorrect_outputs))
        elif 'logiqa' in task.lower():
            if majority_prediction != label:
                # Out of agreement failure - majority prediction is wrong
    #           pdb.set_trace()
                ooa_failed_cases.append((data, incorrect_outputs))

            else:
                # All predictions correct
                correct_cases.append((data, predicted_outputs))

    return ooa_failed_cases, im_failed_cases, correct_cases
def process_validation_batch_count(model_path, data_batch,task,seed=True,  iteration=5):
    """Main function to process NLI validation batch"""
    batch_prompts = [data['Input'] for data in data_batch]
    data_count=[]
    outputs = load_and_generate_predictions(model_path, batch_prompts,seed=seed, iteration=iteration)
    
    # Get model predictions
    #    pdb.set_trace()    
    # Initialize result containers
    
    for id, data in enumerate(data_batch):
        label, majority_prediction, incorrect_outputs, predicted_outputs,predictions = process_single_example(
            data, outputs, id,task, 
        )
#        pdb.set_trace()
        # Categorize results
        if 'nli' in task.lower():
            count=0
            for pred in predictions:
                if pred is not None:
                    if pred==label:
                        count+=1
            data_count.append((data,count/len(predictions)))
        elif 'logiqa' in task.lower():
            count=0
            for pred in predictions:
                if pred is not None:
                    if pred==label:
                        count+=1
            data_count.append((data,count/len(predictions)))
        elif 'gsm8k' in task.lower():
            count=0
            for pred in predictions:
                if pred is not None:
                    if not safty_float_check(pred,label):
                        count+=1
            data_count.append((data,count/len(predictions)))
            
    return data_count
