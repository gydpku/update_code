QA_TEMPLATE = """\
    你是一个口腔修复学领域的专家，负责解答口腔修复学试题。请基于给定的参考信息进行分析。

    ## 相关知识补充
    {context_str}

    ## 问题
    {query_str}

    请优先基于上下文中的知识，结合知识和推理，进行全面、严谨、有根据的思考，给出思考过程与最终答案。最终答案应当只选择一个选项。
    """
ICL_TEMPLATE = """\
    你是一个口腔修复学领域的专家，负责解答口腔修复学试题。请基于给定的参考信息进行分析。

    ## 相关知识补充
    {context_str}

    ## 参考解题思路
    {random_str}

    ## 问题
    {query_str}

    请优先基于上下文中的知识，参考解题思路，进行全面、严谨、有根据的思考，给出思考过程与最终答案。最终答案应当只选择一个选项。
    """

import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import random
from openai import OpenAI
from llama_index.core import Settings, StorageContext, QueryBundle, PromptTemplate

import json


def get_answer(question):
    global client

    completion = client.chat.completions.create(
        model="deepseek-reasoner",
        # model="o1-preview-2024-09-12",
        messages=[
            {
                "role": "user",
                "content": question.strip()
            }
        ]
    )
    return completion.choices[0].message.content

def build_prompt_template(qa_template):
    return PromptTemplate(qa_template)

qa_template = build_prompt_template(QA_TEMPLATE)
icl_template = build_prompt_template(ICL_TEMPLATE)

def process_line(line):
    data = json.loads(line.strip())
    contents = data["rag"]
    context_str = "\n\n".join(
        [f"### 文档{i}: {content}" for i, content in enumerate(contents)]
    )
    query_str = data["query"]
    fmt_qa_prompt = qa_template.format(
        context_str=context_str, query_str=query_str
    )
    answer = get_answer(fmt_qa_prompt)
    return answer

def generation(input_file, output_file):

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    answers = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        # 提交任务到线程池
        futures = [executor.submit(process_line, line) for line in lines]
        
        # 使用 tqdm 显示进度条
        for future in tqdm(futures, total=len(lines), desc="Processing"):
            answers.append(future.result())
    
    if True:
        p=os.path.dirname(output_file)
        if not os.path.exists(p):
            os.makedirs(p)
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, answer in enumerate(answers):
            json_item = {
                "answer": answer
            }
            # 写入jsonl格式（每行一个JSON对象）
            f.write(json.dumps(json_item, ensure_ascii=False) + '\n')


random.seed(42)

paths = [
    "a1/val.jsonl",
    "a1/test.jsonl",
    "a3/val.jsonl",
    "a3/test.jsonl",
]

for shot in [0]:
    for path in paths:
        # 路径到你的JSON文件
        file_path1 = os.path.join("/data/wza/新数据/rag/", path)
        file_path2 = os.path.join(f"/data/wza/新数据/{shot}shot/deepseek", path)

        # 比较键
        generation(file_path1, file_path2)

        # generation_icl(file_path1, file_path2, shot)
