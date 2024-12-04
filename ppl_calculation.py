from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pdb
from tqdm import tqdm
def sort_by_perplexity(data, model_path, batch_size=8):
    """
    Sort examples based on their perplexity scores from the model with optimized batch processing
    Args:
        data: List of examples
        model_path: Path to the model
        batch_size: Size of batches for processing
    Returns:
        Sorted list of examples
    """
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = model.to(device)
    model.eval()

    scores = []
    
    # Process in batches
    for i in tqdm(range(0, len(data), batch_size)):
        batch = data[i:i + batch_size]
        batch_str=[str(data) for data in batch]
        
        # Tokenize batch
        inputs = tokenizer(batch_str, 
                         return_tensors="pt", 
                         padding=True, 
                         truncation=True,
                         max_length=512)
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Calculate loss for each sequence in batch
            loss = outputs.loss
            
            # Calculate perplexity for each sequence
            for j, example in enumerate(batch):
                # Get attention mask for this sequence
                mask = inputs['attention_mask'][j]
                seq_len = mask.sum().item()
                
                pdb.set_trace()                # Calculate per-token loss and perplexity
                ppl = torch.exp(loss * seq_len / mask.sum()).item()
                scores.append((ppl, example))
        
        # Clear CUDA cache periodically
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Clean up
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return scores    
    # Sort by perplexity (lower is better)
    #sorted_data = [item[1] for item in sorted(scores, key=lambda x: x[0])]
    #return sorted_data
