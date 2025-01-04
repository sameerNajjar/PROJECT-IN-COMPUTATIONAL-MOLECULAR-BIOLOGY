import os
from Bio import SeqIO
from Bio.Align import PairwiseAligner
import random
import torch
import numpy as np
from transformers import EsmTokenizer, EsmModel

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
### get per residue ESM2 embeddings
###########################################
def get_all_per_residue_embeddings(sequence, tokenizer, model, device="cpu"):
    inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)  
    per_residue_embs = outputs.last_hidden_state[0]
    return per_residue_embs  

###########################################
### cosine similarity-based scoring
###########################################
def distance_score(emb1, emb2):
    cosine_similarity = torch.dot(emb1, emb2) / (torch.norm(emb1) * torch.norm(emb2))
    return cosine_similarity.item()

################################################
### negative L2 distance based similarity
################################################
def distance_score2(emb1, emb2):
    dist = torch.norm(emb1 - emb2, p=2)
    return -dist.item()

########################################################
### Needleman-Wunsch alignment for embeddings
########################################################
def needleman_wunsch_align(seq1_embs, seq2_embs, scoring_func, gap_penalty=-5.0):
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
            aligned_indices.append((None, j - 1))
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
    dataset_path="balibase_dataset/RV100"
    fasta_file= extract_sequences(dataset_path)
    fasta_file = "extracted_sequences.fasta"
    model_name = "facebook/esm2_t30_150M_UR50D"
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    seq_records = list(SeqIO.parse(fasta_file, "fasta"))
    if len(seq_records) < 2:
        print("Need at least 2 sequences for alignment. Exiting.")
        return

    seq1 = str(seq_records[0].seq)
    seq2 = str(seq_records[1].seq)
    print(seq1)
    print(seq2)
    print(f"Sequence 1: {seq_records[0].id}, length={len(seq1)}")
    print(f"Sequence 2: {seq_records[1].id}, length={len(seq2)}")

    seq1_embs = get_all_per_residue_embeddings(seq1, tokenizer, model, device)
    seq2_embs = get_all_per_residue_embeddings(seq2, tokenizer, model, device)
    seq1_embs = seq1_embs / torch.norm(seq1_embs, dim=-1, keepdim=True)
    seq2_embs = seq2_embs / torch.norm(seq2_embs, dim=-1, keepdim=True)

    dtw_score, dtw_indices = needleman_wunsch_align(seq1_embs, seq2_embs, distance_score)

    aligned_seq1 = []
    aligned_seq2 = []
    for (i, j) in dtw_indices:
        aligned_seq1.append(seq1[i] if i is not None else '-')
        aligned_seq2.append(seq2[j] if j is not None else '-')

    aligned_seq1_str = ''.join(aligned_seq1)
    aligned_seq2_str = ''.join(aligned_seq2)

    identities, positives, gaps, alignment_length = calculate_statistics(aligned_seq1_str, aligned_seq2_str)

    print("Alignment Statistics:")
    print(f"Identities: {identities}/{alignment_length} ({identities / alignment_length * 100:.2f}%)")
    print(f"Substitutions: {positives - identities}/{alignment_length} ({(positives - identities) / alignment_length * 100:.2f}%)")
    print(f"Gaps: {gaps}/{alignment_length} ({gaps / alignment_length * 100:.2f}%)")

if __name__ == "__main__":
    main()
