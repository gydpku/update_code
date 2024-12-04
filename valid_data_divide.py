import gc
import pdb
import torch
import time
from collections import Counter
from vllm import LLM, SamplingParams
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

def process_single_example(data, outputs, id, labels):
    """Process predictions for a single example"""
    label = extract_answer(data['Output'].split('is')[-1])
    predictions = []
    predicted_outputs = []
    
    # Collect predictions
    for output in outputs:
        predicted_output = output[id].outputs[0].text
        predicted_res = extract_answer_prediction_nli(predicted_output)
        if not predicted_res:
            predicted_res = extract_answer(predicted_output)
        predictions.append(predicted_res)
        predicted_outputs.append(predicted_output)
    
    # Get incorrect predictions
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
#    incorrect_outputs = [
 #       output.split("Let's think step by step.")[1] for output, pred in zip(predicted_outputs, predictions)
  #      if pred != label
  #  ]
    
    # Calculate majority vote
    prediction_counts = Counter(predictions)
    majority_prediction = prediction_counts.most_common(1)[0][0]
#    pdb.set_trace()    
    return label, majority_prediction, i_o, predicted_outputs,predictions

def process_nli_validation_batch(model_path, data_batch,seed=True, iteration=5,temperature=0.2):
    """Main function to process NLI validation batch"""
    batch_prompts = [data['Input'] for data in data_batch]
    labels = ['entailment', 'contradiction', 'neutral']
    
    # Get model predictions
    outputs = load_and_generate_predictions(model_path, batch_prompts,seed=seed, iteration=iteration,temperature=temperature)
#    pdb.set_trace()    
    # Initialize result containers
    ooa_failed_cases = []  # out-of-agreement failures
    im_failed_cases = []   # inconsistent-majority failures
    correct_cases = []
    
    # Process each example
    for id, data in enumerate(data_batch):
        label, majority_prediction, incorrect_outputs, predicted_outputs,predictions = process_single_example(
            data, outputs, id, labels
        )
#        pdb.set_trace()
        # Categorize results
        if label not in predictions: #majority_prediction != label:
            # Out of agreement failure - majority prediction is wrong
 #           pdb.set_trace()
            ooa_failed_cases.append((data, incorrect_outputs))
        elif incorrect_outputs:  # len(incorrect_outputs) > 0
            # Inconsistent majority failure - majority is correct but some predictions wrong
            im_failed_cases.append((data, incorrect_outputs))
        else:
            # All predictions correct
            correct_cases.append((data, predicted_outputs))
    
    return ooa_failed_cases, im_failed_cases, correct_cases
def process_nli_validation_batch_major(model_path, data_batch,seed=True, iteration=5):
    """Main function to process NLI validation batch"""
    batch_prompts = [data['Input'] for data in data_batch]
    labels = ['entailment', 'contradiction', 'neutral']
    
    # Get model predictions
    outputs = load_and_generate_predictions(model_path, batch_prompts,seed=seed, iteration=iteration)
#    pdb.set_trace()    
    # Initialize result containers
    ooa_failed_cases = []  # out-of-agreement failures
    im_failed_cases = []   # inconsistent-majority failures
    correct_cases = []
    for id, data in enumerate(data_batch):
        label, majority_prediction, incorrect_outputs, predicted_outputs,predictions = process_single_example(
            data, outputs, id, labels
        )
#        pdb.set_trace()
        # Categorize results
        if majority_prediction != label:
            # Out of agreement failure - majority prediction is wrong
 #           pdb.set_trace()
            ooa_failed_cases.append((data, incorrect_outputs))
        #elif incorrect_outputs:  # len(incorrect_outputs) > 0
            # Inconsistent majority failure - majority is correct but some predictions wrong
        #    im_failed_cases.append((data, incorrect_outputs))
        else:
            # All predictions correct
            correct_cases.append((data, predicted_outputs))
    
    return ooa_failed_cases, im_failed_cases, correct_cases
