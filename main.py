import os
from Bio import SeqIO
import random
import torch
import numpy as np
from transformers import EsmTokenizer, EsmModel
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein

###############################################################
#### extract sequences randomly from the dataset 
###############################################################
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

###########################################
### get per residue embeddings using esmc
###########################################
def get_esmc_embeddings(sequence, model, device):
    protein = ESMProtein(sequence=sequence)
    input_ids = model._tokenize([protein.sequence]).to(device)  
    output = model(input_ids)
    embeddings = output.embeddings[0, 1:-1]  
    return embeddings.to(device) 

###########################################
### get per residue embeddings using esm2
###########################################
def get_esm2_embeddings(sequence, tokenizer, model, device):
    inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(device) for k, v in inputs.items()} 
    with torch.no_grad():
        outputs = model(**inputs)
    per_residue_embs = outputs.last_hidden_state[0]
    return per_residue_embs.to(device)

###########################################
### cosine similarity-based scoring
###########################################
def distance_score(emb1, emb2):
    cosine_similarity = torch.dot(emb1, emb2) / (torch.norm(emb1) * torch.norm(emb2))
    return cosine_similarity.item()

def euclidean_distance_score(emb1, emb2):
    return -torch.norm(emb1 - emb2, p=2).item()
########################################################
### Needleman-Wunsch alignment for embeddings
########################################################
def needleman_wunsch_align(seq1_embs, seq2_embs, scoring_func, gap_penalty=-10.0):
    L1 = seq1_embs.shape[0]
    L2 = seq2_embs.shape[0]

    dp = np.zeros((L1 + 1, L2 + 1), dtype=np.float32)
    dp[:, 0] = np.arange(L1 + 1) * gap_penalty
    dp[0, :] = np.arange(L2 + 1) * gap_penalty

    traceback = np.zeros((L1 + 1, L2 + 1), dtype=int)

    for i in range(1, L1 + 1):
        for j in range(1, L2 + 1):
            match = dp[i - 1, j - 1] + scoring_func(seq1_embs[i - 1], seq2_embs[j - 1])
            delete = dp[i - 1, j] + gap_penalty
            insert = dp[i, j - 1] + gap_penalty
            dp[i, j] = max(match, delete, insert)
            traceback[i, j] = np.argmax([match, delete, insert])

    i, j = L1, L2
    aligned_indices = []

    while i > 0 or j > 0:
        if traceback[i, j] == 0:  # Match
            aligned_indices.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif traceback[i, j] == 1:  # Deletion
            aligned_indices.append((i - 1, None))
            i -= 1
        else:  # Insertion
            j -= 1

    aligned_indices.reverse()
    return dp[L1, L2], aligned_indices

########################################################
### Count matches, positives, and gaps
########################################################
def calculate_statistics(aligned_seq1, aligned_seq2):
    identities = 0
    positives = 0
    gaps = 0

    for a, b in zip(aligned_seq1, aligned_seq2):
        if a == '-' or b == '-':
            gaps += 1
        elif a == b:
            identities += 1
            positives += 1  
        else:
            positives += 1  
    alignment_length = len(aligned_seq1)
    return identities, positives, gaps, alignment_length

########################################################
### Main function
########################################################
def main():
    dataset_path = "balibase_dataset/RV100"
    fasta_file = extract_sequences(dataset_path)
    fasta_file = "extracted_sequences.fasta"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    esm2_model_name = "facebook/esm2_t33_650M_UR50D"
    esm2_tokenizer = EsmTokenizer.from_pretrained(esm2_model_name)
    esm2_model = EsmModel.from_pretrained(esm2_model_name)
    esm2_model.to(device) 
    
    esmc_model_name = "esmc_600m"
    esmc_model = ESMC.from_pretrained(esmc_model_name)
    esmc_model.to(device)  

    seq_records = list(SeqIO.parse(fasta_file, "fasta"))
    if len(seq_records) < 2:
        print("Need at least 2 sequences for alignment. Exiting.")
        return

    seq1 = str(seq_records[0].seq)
    seq2 = str(seq_records[1].seq)

    print(f"Sequence 1: {seq_records[0].id}, length={len(seq1)}")
    print(f"Sequence 2: {seq_records[1].id}, length={len(seq2)}")

    esm2_seq1_embs = get_esm2_embeddings(seq1, esm2_tokenizer, esm2_model, device)
    esm2_seq2_embs = get_esm2_embeddings(seq2, esm2_tokenizer, esm2_model, device)

    esmc_seq1_embs = get_esmc_embeddings(seq1, esmc_model, device)
    esmc_seq2_embs = get_esmc_embeddings(seq2, esmc_model, device)

    esm2_seq1_embs = esm2_seq1_embs / torch.norm(esm2_seq1_embs, dim=-1, keepdim=True)
    esm2_seq2_embs = esm2_seq2_embs / torch.norm(esm2_seq2_embs, dim=-1, keepdim=True)

    esmc_seq1_embs = esmc_seq1_embs / torch.norm(esmc_seq1_embs, dim=-1, keepdim=True)
    esmc_seq2_embs = esmc_seq2_embs / torch.norm(esmc_seq2_embs, dim=-1, keepdim=True)

    esm2_score, esm2_indices = needleman_wunsch_align(esm2_seq1_embs, esm2_seq2_embs, euclidean_distance_score)
    esm2_aligned_seq1 = ''.join(seq1[i] if i is not None else '-' for i, _ in esm2_indices)
    esm2_aligned_seq2 = ''.join(seq2[j] if j is not None else '-' for _, j in esm2_indices)
    esm2_identities, esm2_positives, esm2_gaps, esm2_length = calculate_statistics(esm2_aligned_seq1, esm2_aligned_seq2)

    esmc_score, esmc_indices = needleman_wunsch_align(esmc_seq1_embs, esmc_seq2_embs, euclidean_distance_score)
    esmc_aligned_seq1 = ''.join(seq1[i] if i is not None else '-' for i, _ in esmc_indices)
    esmc_aligned_seq2 = ''.join(seq2[j] if j is not None else '-' for _, j in esmc_indices)
    esmc_identities, esmc_positives, esmc_gaps, esmc_length = calculate_statistics(esmc_aligned_seq1, esmc_aligned_seq2)

    print("\nESM2 Alignment Statistics:")
    print(f"Identities: {esm2_identities}/{esm2_length} ({esm2_identities / esm2_length * 100:.2f}%)")
    print(f"Substitutions: {esm2_positives - esm2_identities}/{esm2_length} ({(esm2_positives - esm2_identities) / esm2_length * 100:.2f}%)")
    print(f"Gaps: {esm2_gaps}/{esm2_length} ({esm2_gaps / esm2_length * 100:.2f}%)")

    print("\nESMC Alignment Statistics:")
    print(f"Identities: {esmc_identities}/{esmc_length} ({esmc_identities / esmc_length * 100:.2f}%)")
    print(f"Substitutions: {esmc_positives - esmc_identities}/{esmc_length} ({(esmc_positives - esmc_identities) / esmc_length * 100:.2f}%)")
    print(f"Gaps: {esmc_gaps}/{esmc_length} ({esmc_gaps / esmc_length * 100:.2f}%)")
    print(f"ESM2 Alignment Score: {esm2_score}")
    print(f"ESMC Alignment Score: {esmc_score}")
    
if __name__ == "__main__":
    main()
