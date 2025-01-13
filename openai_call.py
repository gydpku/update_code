import openai
import anthropic
import tiktoken
import json
import os
import pdb
from openai import OpenAI
#from vllm import LLM, SamplingParams
from datasets import load_dataset
# Replace this with your actual OpenAI API key
#sampling_params = SamplingParams(temperature=0.0,max_tokens=100, top_p=0.95)
#llm = LLM(model='/dccstor/obsidian_llm/yiduo/llama-3-instruct',gpu_memory_utilization=0.8)
encoding = tiktoken.encoding_for_model("gpt-4")


def truncate_text_with_token_count(text, max_tokens):
    num_tokens = len(encoding.encode(text))
    if num_tokens > max_tokens:
        encoded = encoding.encode(text)[:-(num_tokens - max_tokens)]
        truncated_text = encoding.decode(encoded)
        return truncated_text
    return text

#def query_azure_openai_chatgpt_chat_2(query, temperature=0):
#    truncated_input = truncate_text_with_token_count(query, 2048)
#    sampling_params = SamplingParams(temperature=0.0,max_tokens=100, top_p=0.95)
 #   llm = LLM(model='/dccstor/obsidian_llm/yiduo/llama-3-instruct',gpu_memory_utilization=0.8)
#    prompt=truncated_input
#    output = llm.generate(prompt, sampling_params)
#    output=output[0].outputs[0].text
#    return output
def query_azure_openai_chatgpt_chat_2(query, temperature=0):
    truncated_input = truncate_text_with_token_count(query, 30000)
    client = OpenAI(api_key=openai.api_key)
    response = client.messages.create(
        model="gpt4-o1",
        max_tokens=4000,
        temperature=temperature,
        messages=[{"role": "user", "content": truncated_input}]
    )
    return response.content[0].text
def o1_chat_completion(prompt, model="o1-preview"):
    messages = [{"role": "user", "content": prompt}]
    client = OpenAI(api_key=openai.api_key)
    response = client.chat.completions.create(model="o1-preview", messages=messages)
    return response
def query_azure_openai_chatgpt_chat(query, temperature=0,n=1):
    truncated_input = truncate_text_with_token_count(query, 30000)

    client = OpenAI(api_key=openai.api_key)

    response = client.chat.completions.create(model="gpt-4o", messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": truncated_input}, ], temperature=temperature, max_tokens=4000,n=n)
    responses = [choice.message.content for choice in response.choices]
    if n==1:
        return responses[0] 
    else:
        return responses
def query_azure_openai_chatgpt_chat_3(query, temperature=0):
    query = truncate_text_with_token_count(query, 30000)
    '''    
    sampling_params = SamplingParams(temperature=0.0,max_tokens=100, top_p=0.95)
    llm = LLM(model='/dccstor/obsidian_llm/yiduo/llama-3-instruct',gpu_memory_utilization=0.8)
    prompt=truncated_input
    output = llm.generate(prompt, sampling_params)
    output=output[0].outputs[0].text
    '''
    client = OpenAI(api_key=openai.api_key)

    response = client.chat.completions.create(model="gpt-4", messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": query}, ], temperature=temperature, max_tokens=4000, )
    #print(temperature)
    for chunk in response:
        if chunk[0]=='choices':
            for piece in chunk[1][0]:
                if piece[0]=='message':
                    for sub in piece[1]:
                        if sub[0]=='content':
                            output=sub[1]
    
    #pdb.set_trace()
    return output
