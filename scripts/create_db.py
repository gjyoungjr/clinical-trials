"""
python scripts/create_db.py --data_path "data/processed_data.csv" --embed_model "medicalai/ClinicalBERT"
"""

from typing import List
from dotenv import load_dotenv
import torch 
import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import load_model, embed

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
    
    print(f"Loading model {embed_model}")
    print(f"Loading data from {data_path}")
    print(f"Delete collection: {is_delete_collection}")
    
    
    # Load Model 
    model, tokenizer = load_model(embed_model)
    print("Device:", model.device)