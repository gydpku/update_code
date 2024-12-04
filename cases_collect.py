from vllm import LLM, SamplingParams
import multiprocessing
import time
import gc
import torch
import pdb
import sqlite3
from concurrent.futures import ThreadPoolExecutor
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
    elif task=='nli':
        failed_cases,correct_cases=nli_evaluation(trained_model,valid_data)
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
        if extract_answer(sen):
            return extract_answer(sen)
    return
def extract_answer(text):
    if 'neutral' in text.lower():
        return 'neutral'
    if 'contradiction' in text.lower():
        return 'contradiction'
    if 'entailment' in text.lower():
        return 'entailment'
    return None
def process_batch(data_batch,trained_model,failed_cases,correct_cases):
    batch_prompts = [data['Input'] for data in data_batch]
    outputs = trained_model.generate(batch_prompts, sampling_params)
    
    results = []
    labels=['entailment','contradiction','neutral']
    for data, output in zip(data_batch, outputs):
#        pdb.set_trace()
        predicted_output = output.outputs[0].text
        predicted_res = extract_answer_prediction_nli(predicted_output)
        label = extract_answer(data['Output'].split('is')[-1])

#        print(predicted_res,label,'\n')
#        pdb.set_trace()
        if not predicted_res:
#            pdb.set_trace()
            
            predicted_res=predicted_output
#            print(label,predicted_output) # if 'contradiction #label_transform(data['Output'])
#        pdb.set_trace()
 #       print(predicted_res,label,'\n')
        non_labels = [lbl for lbl in labels if lbl != label]
        if label not in predicted_res or any(non_label in predicted_res for non_label in non_labels):
            failed_cases.append((data['Input'],predicted_res,label,data))
        else:
            correct_cases.append((data['Input'],predicted_res,label,data))
    return failed_cases,correct_cases
def nli_evaluation(trained_model,valid_data):
    id=0
    failed_cases=[]
    correct_cases=[]
    batch_size=500
    batched_data = [valid_data[i:i+batch_size] for i in range(0, len(valid_data), batch_size)]
    for batch in batched_data:
        failed_cases,correct_cases=process_batch(batch,trained_model,failed_cases,correct_cases)
        
    #for data in valid_data:
    #    prompt=data['Input']
    #    output=trained_model.generate(prompt, sampling_params)
    #    predicted_output=output[0].outputs[0].text
    #    predicted_res=extract_answer_prediction_nli(predicted_output) #$try:
    #    #    predicted_res=extract_answer(predicted_output.split('final')[-1].split('is')[1].split('.')[0])
        #except:
        #    predicted_res=extract_answer(predicted_output.split('is')[-1])
   #     label=extract_answer(data['Output'].split('is')[-1])
   #     print(label,predicted_res)
   #     if not predicted_res:
   #         pdb.set_trace()
   #         predicted_res=''
       # if 'contradiction #label_transform(data['Output'])
#        pdb.set_trace()
   #     if label not in predicted_res:
   #         failed_cases.append((id,prompt,predicted_res,label,data))
   #     else:
   #         correct_cases.append((id,prompt,predicted_res,label,data))
   #     id+=1
    #id,prompt,prior_pred+predicted_sql,valid_data[id],ground_truth,predicted_res,ground_truth_res
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
