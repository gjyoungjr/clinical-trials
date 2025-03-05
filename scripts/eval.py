"""
python scripts/eval.py --llm_model "gpt-4o-mini" --embed_model "medicalai/ClinicalBERT" --n_chunks 9999
"""

import os 
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import get_relevant_docs_for_physicians, load_model, load_chroma_collection


physician_profile = {
      "id": "P001",
      "name": "Dr. Alice Thompson",
      "medical_specialties": ["Gynecologic Oncology", "Medical Oncology"],
      "technical_skills": [
        "Immunotherapy",
        "Targeted Therapy",
        "Biomarker Analysis"
      ],
      "years_of_experience": {
        "Gynecologic Oncology": 15,
        "Clinical Research": 12,
        "Precision Medicine": 8
      },
      "research_interests": [
        "PARP inhibitors",
        "BRCA mutations",
        "Immunotherapy advancements"
      ]
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate the model on the given data')
    parser.add_argument('--embed_model', type=str, default='sentence-transformers/all-MiniLM-L6-v2', help='Name of the model to use')
    parser.add_argument('--llm_model', type=str, default='gpt-3.5-turbo-1106', help='Name of the LLM to use for criteria assessment')
    parser.add_argument('--n_chunks', type=int, default=None, help='# of most similar chunks to use. Use `9999` to return everything')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    llm_model: str = args.llm_model
    embed_model: str = args.embed_model
    
    # Load embed model 
    model, tokenizer = load_model(embed_model)
    
    print(f"Chroma collection name: {embed_model.split('/')[-1]}")

    # Load chroma collection 
    chroma_collection = load_chroma_collection(embed_model.split("/")[-1])
    
    # clean up phsycian profile
    experience_text = "; ".join(
    f"{area}: {years} years" for area, years in physician_profile["years_of_experience"].items()
) 
    physician_profile_text = f"""
    {physician_profile['name']} is a distinguished physician specializing in {', '.join(physician_profile['medical_specialties'])}. 
    With extensive experience in the field,  {physician_profile['name']} has worked in various areas including {experience_text}. {physician_profile['name']} is highly skilled in {', '.join(physician_profile['technical_skills'])}, applying these techniques in cutting-edge clinical trials for ovarian cancer. Her research focuses on {', '.join(physician_profile['research_interests'])}, contributing significantly to advancements in targeted treatments and precision medicine. With a commitment to innovation and patient-centered care,  {physician_profile['name']} continues to push the boundaries of medical oncology, striving for better outcomes in ovarian cancer treatment.
    """
    
    # Get relevant trials for physicians
    clinical_trials = get_relevant_docs_for_physicians(physician_profile=physician_profile_text,
                                    collection=chroma_collection,
                                    model=model,
                                    tokenizer=tokenizer,
                                    n_chunks=args.n_chunks,
                                    threshold=None)
    print(clinical_trials)
    
    
    