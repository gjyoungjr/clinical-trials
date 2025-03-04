
import os
import torch
from transformers import AutoModel, AutoTokenizer
from typing import Tuple

def load_model(model_name: str) -> Tuple[AutoModel, AutoTokenizer]:
    """Load HF model and tokenizer"""
    
    os.environ['TOKENIZERS_PARALLELISM'] = 'False'
    device: str = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    
    if model_name in ['medicalai/ClinicalBERT']:
        model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16, use_flash_attention_2=True)
    else:
        model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to(device)
    
    return model, tokenizer


def embed(text: str, model: AutoModel, tokenizer: AutoTokenizer) -> torch.Tensor:
    """Embeds text using the given model and tokenizer"""
    device: str = model.device
    tokens: torch.Tensor = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
    embedding: torch.Tensor = model(**tokens)[0][0].mean(dim=0).detach().cpu()
    embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)
    return embedding