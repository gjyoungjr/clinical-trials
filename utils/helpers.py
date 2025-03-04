import os
import torch
from transformers import AutoModel, AutoTokenizer
from typing import Tuple
import pandas as pd

def load_model(model_name: str) -> Tuple[AutoModel, AutoTokenizer]:
    """Load HF model and tokenizer"""
    
    os.environ['TOKENIZERS_PARALLELISM'] = 'False'
    device: str = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    
    ## TODO: Change to 'emilyalsentzer/Bio_ClinicalBERT', check eval for better performance
    if model_name in ['medicalai/ClinicalBERT']:
        model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
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



def clean_dataset(data_path: str) -> str:
    """Convert dataset into readbable text for better understanding"""
    df = pd.read_csv(data_path)
        
    df["full_text"] = df.apply(lambda row: 
    f"The clinical trial titled '{row['Study Title']}' is currently {row['Study Status']}. "
    f"This {row['Study Type'].lower()} study started on {row['Start Date']} and is expected to end on {row['Completion Date']}. "
    f"It is investigating {row['Conditions']} and involves the intervention(s): {row['Interventions']}. "
    f"Primary outcome: {row['Primary Outcome Measures']}. "
    f"Brief Summary: {row['Brief Summary']}.",
    axis=1
    )
    texts = df["full_text"].tolist()
    
    return texts
