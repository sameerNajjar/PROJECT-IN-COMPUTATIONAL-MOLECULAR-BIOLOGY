import os
from Bio import SeqIO
import random
import torch
from transformers import EsmTokenizer, EsmModel

def extract_sequences(dataset_path, num_sequences=100, output_file='extracted_sequences.fasta'):
    all_sequences = []
    
    for filename in os.listdir(dataset_path):
        if filename.endswith('.tfa'):
            file_path = os.path.join(dataset_path, filename)
            
            try:
                file_sequences = list(SeqIO.parse(file_path, 'fasta'))
                all_sequences.extend(file_sequences)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    
    if len(all_sequences) < num_sequences:
        print(f"Only {len(all_sequences)} sequences found.")
        selected_sequences = all_sequences
    else:
        selected_sequences = random.sample(all_sequences, num_sequences)
    
    SeqIO.write(selected_sequences, output_file, 'fasta')
    
    return selected_sequences

# https://huggingface.co/docs/transformers/model_doc/esm
# https://huggingface.co/facebook/esm2_t36_3B_UR50D
def main():
    fasta_file = "extracted_sequences.fasta"
    model_name = "facebook/esm2_t30_150M_UR50D"
    
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    for seq_record in SeqIO.parse(fasta_file, "fasta"):
        sequence = str(seq_record.seq)
        inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        embeddings = outputs.last_hidden_state
        sequence_embeddings = embeddings[:, 1:-1, :].mean(dim=1)  
        
        print(f"ID: {seq_record.id}, shape: {sequence_embeddings.shape}")
        
if __name__ == "__main__":
    main()