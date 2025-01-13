from rank_bm25 import BM25Okapi
from tqdm import tqdm #import tqdm
import os
import json
from openai_call import query_azure_openai_chatgpt_chat

def search_relevant_texts(texts, keywords,top_k=10000):
    """
    Search for relevant texts based on keywords using BM25.

    Args:
        texts (list of str): List of documents (texts) to search.
        keywords (list of str): List of keywords to match against the documents.

    Returns:
        list of tuples: Sorted list of tuples where each tuple contains a document and its BM25 score.
    """
    # Tokenizing function
    def tokenize(text):
        return text.lower().split()

    # Tokenize the texts and keywords
    tokenized_texts = [tokenize(text) for text in tqdm(texts, desc="Processing tokenizeing")]
    tokenized_keywords = tokenize(" ".join(keywords))

    # Create BM25 object
    bm25 = BM25Okapi(tokenized_texts)

    # Get BM25 scores for the keywords (query)
    scores = bm25.get_scores(tokenized_keywords)

    # Sort documents based on BM25 score
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    sorted_documents = [texts[i] for i in sorted_indices[:top_k]]

    return sorted_documents

def collect_relevant_texts(paths=['/dccstor/obsidian_llm/yiduo/datasets/stackexchange/cleaned','/dccstor/obsidian_llm/yiduo/datasets/wikihow/cleaned','/dccstor/obsidian_llm/yiduo/datasets/wikipedia/cleaned']):
 #/dccstor/obsidian_llm/yiduo/datasets/wikipedia/cleaned
 #/dccstor/obsidian_llm/yiduo/datasets/wikihow/cleaned
 #/dccstor/obsidian_llm/yiduo/datasets/stackexchange/cleaned
    texts = []

    # Outer loop to iterate through paths
    for path in tqdm(paths, desc="Processing directories"):
        jsonl_paths = [f for f in os.listdir(path) if f.endswith('.jsonl')]
        # Inner loop for JSONL files
        for jsonl_path in tqdm(jsonl_paths, desc=f"Processing files in {path}", leave=False):
            with open(os.path.join(path, jsonl_path), 'r') as file:
                for line in file:
                    data = json.loads(line)
                    try:
                        texts.append(data['text'])
                    except:
                        texts.append(' '.join([v for k,v in data.items()]))
    
    return texts

def extract_keywords(domain_name):
    def prompt_synonyms(text):
        return 'You can generate synonyms about this domain:{0}. Each synonym must have the same meaning as the domain. Your output must only be a list of all synonyms like ["xxx","xxx"].'.format(text)
    def extract_list(text):
        index1=text.find('[')
        index2=text.find(']')
        return eval(text[index1:index2+1])
    domain_syns=query_azure_openai_chatgpt_chat(prompt_synonyms(domain_name)) 
    domain_syns=extract_list(domain_syns)
    return domain_syns

def search_relevant_documents(domain_name,top_k,paths=['/dccstor/obsidian_llm/yiduo/datasets/stackexchange/cleaned','/dccstor/obsidian_llm/yiduo/datasets/wikihow/cleaned','/dccstor/obsidian_llm/yiduo/datasets/wikipedia/cleaned']):
    domain_syns=extract_keywords(domain_name)
    print("Keywords",domain_syns)
    texts=collect_relevant_texts(paths)
    print("text",len(texts))
    relevant_texts=search_relevant_texts(texts,domain_syns,top_k=top_k)
    return relevant_texts
