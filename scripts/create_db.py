"""
python scripts/create_db.py --data_path "data/ovarian_cancer.csv" --embed_model "medicalai/ClinicalBERT"
"""

from typing import List
from dotenv import load_dotenv
import torch 
import argparse
import sys
import os
from tqdm import tqdm
import chromadb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import load_model, clean_dataset

   
def store_embeddings(model, data: List[dict], chromadb_collection, batch_size=30):
    """Generate embeddings for the given data and store in the collection"""    
    for trial in tqdm(range(0, len(data), batch_size), desc="Generating Embeddings"):
        batch_data = data[trial: trial + batch_size]  # Get batch
        
        batch_texts = [entry['text'] for entry in batch_data]
        model_max_len: int = model.config.max_position_embeddings
        
        # Tokenize the batch of texts
        tokens = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)['input_ids'].to(model.device)
        
        for chunk_idx, chunk_start in enumerate(range(0, len(tokens[0]), model_max_len)):
            embedding: torch.Tensor = model(input_ids=tokens[:, chunk_start:chunk_start + model_max_len])[0][0].mean(dim=0).detach().cpu()
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)
            
            for idx, entry in enumerate(batch_data):
                metadata = entry['metadata']  
                
                print(f"Upserting embedding for chunk {chunk_idx}: {embedding}")              
                # Upsert the embedding and metadata into chromadb_collection
                chromadb_collection.upsert(
                    embeddings=embedding.tolist(),
                    metadatas={
                        'chunk_idx': chunk_idx,
                        'chunk_start': chunk_start,
                        'chunk_end': chunk_start + model_max_len,
                        **metadata  # Merge the current entry's metadata
                    },
                    documents=tokenizer.decode(tokens[0,chunk_start:chunk_start+model_max_len].tolist()),
                    ids=f"{chunk_idx}",
                )
                
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create embeddings for the given text")
    parser.add_argument("--data_path", type=str, help="Path to the data file")
    parser.add_argument("--embed_model", type=str, help="Name of the model to use")
    parser.add_argument("--is_delete_collection", action="store_true", default=False, help='If TRUE, clear out collection (if already exists).')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    embed_model: str = args.embed_model
    data_path: str = args.data_path
    is_delete_collection: bool = args.is_delete_collection   
    
    # Setup chroma db 
    collection_name: str = embed_model.split("/")[-1]
    client = chromadb.PersistentClient(path="./data/chroma")
    if is_delete_collection and client.get_collection(collection_name) is not None:
        print(f"Deleting existing collection: `{collection_name}`...")
        client.delete_collection(collection_name)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    
    # Load & clean dataset
    dataset = clean_dataset(data_path) 
    print(f"Number of clinical trials: {len(dataset)}")
        
    # Load Model 
    model, tokenizer = load_model(embed_model)
    print("Device:", model.device)
    print("Chunk size:", model.config.max_position_embeddings)
    
    store_embeddings(model=model, data=dataset, chromadb_collection=collection)
