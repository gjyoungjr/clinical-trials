import os
import torch
from transformers import AutoModel, AutoTokenizer
from typing import Tuple
import pandas as pd
from typing import Dict, List, Any, Tuple, Union, Optional
import chromadb

PATH_TO_CHROMADB = "data/chroma"

def load_chroma_collection(collection_name: str) -> chromadb.Collection:
    """Load Chroma db collection from disk"""
    path_to_chromadb: str = os.environ.get("PATH_TO_CHROMADB", PATH_TO_CHROMADB)
    client = chromadb.PersistentClient(path=path_to_chromadb)
    collection = client.get_collection(collection_name)
    return collection

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


def clean_dataset(data_path: str) -> list:
    """Convert dataset into readable text with metadata for better understanding"""
    df = pd.read_csv(data_path)
    
    # Function to create the text and return a dictionary with metadata
    def create_text_and_metadata(row):
        text = (f"The clinical trial titled '{row['Study Title']}' is currently {row['Study Status']}. "
                f"This {row['Study Type'].lower()} study started on {row['Start Date']} and is expected to end on {row['Completion Date']}. "
                f"It is investigating {row['Conditions']} and involves the intervention(s): {row['Interventions']}. "
                f"Primary outcome: {row['Primary Outcome Measures']}. "
                f"Brief Summary: {row['Brief Summary']}.")
        
        # Return a dictionary with full text and metadata
        metadata = {
            "study_type": row['Study Type'],
            "conditions": row['Conditions'],
            "interventions": row['Interventions'],
            
        }
        
        return {"text": text, "metadata": metadata}
    
    # Apply the function to each row and collect the results
    results = df.apply(create_text_and_metadata, axis=1).tolist()
    
    return results

def get_relevant_docs_for_physicians(physician_profile: str, collection: chromadb.Collection, 
                                     model: AutoModel, tokenizer: AutoTokenizer, n_chunks: Optional[int], 
                                     threshold: Optional[float]) -> List[Dict[str, Any]]:
    """
        Given a physician's profile, query Chroma db for relevant documents for that physician.
        Return the top-n_chunks
    """
    

    assert n_chunks is None or n_chunks > 0, "n_chunks must be None or > 0"
    assert threshold is None or (threshold >= 0 and threshold <= 1), "threshold must be None or between 0 and 1"
    assert n_chunks is None or threshold is None, "n_chunks and threshold cannot both be specified"
    
    # Embed physician profile
    physician_profile_embedding = embed(physician_profile, model, tokenizer).tolist()

    # Query Chroma
    # TODO: Use the args to run conditional queries
    results: Dict = collection.query(
            query_embeddings=physician_profile_embedding,
            include=["metadatas", "documents", "distances", ],
            n_results=10,
        )
    
    print(results)
     # Filter results to only those >= similarity threshold
    # records: List[Dict[str, Any]] = []
    # for id, distance, text, metadata in zip(results['ids'][0], results['distances'][0], results['documents'][0], results['metadatas'][0]):
    #     similarity: float = 1 - distance
    #     if threshold is not None:
    #         if similarity < threshold:
    #             continue
    #     if len(text) < 40:
    #         # Ignore chunks < 40 chars b/c meaningless
    #         continue
    #     records.append({
    #         'id' : id,
    #         'metadata' : metadata,
    #         'similarity' : similarity,
    #         'text' : text,
    #     })

    # return sorted(records, key=lambda x: (int(x['metadata']['chunk_idx']))) # sort by note_idx, then chunk_idx