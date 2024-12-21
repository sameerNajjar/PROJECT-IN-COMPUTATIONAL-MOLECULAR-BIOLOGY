import os
from Bio import SeqIO
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

################################################
### negative L2 distance based similarity
################################################
def distance_score(emb1, emb2):
    dist = torch.norm(emb1 - emb2, p=2)
    return -dist.item()

######################################################################################################
### needleman-wunsch with distance score according the per residue ESM2 embeddings
######################################################################################################
def needleman_wunsch_align(seq1_embs, seq2_embs, scoring_func, gap_penalty=-5.0):
    L1 = seq1_embs.shape[0]
    L2 = seq2_embs.shape[0]
    dp = np.zeros((L1+1, L2+1), dtype=np.float32)
    traceback = np.zeros((L1+1, L2+1), dtype=np.int32)  
    
    for i in range(1, L1+1):
        dp[i, 0] = dp[i-1, 0] + gap_penalty
        traceback[i, 0] = 1  
    for j in range(1, L2+1):
        dp[0, j] = dp[0, j-1] + gap_penalty
        traceback[0, j] = 2 

    for i in range(1, L1+1):
        for j in range(1, L2+1):
            match_score = dp[i-1, j-1] + scoring_func(seq1_embs[i-1], seq2_embs[j-1])

            gap_in_seq2 = dp[i-1, j] + gap_penalty
            gap_in_seq1 = dp[i, j-1] + gap_penalty

            best_score = max(match_score, gap_in_seq2, gap_in_seq1)
            dp[i, j] = best_score

            if best_score == match_score:
                traceback[i, j] = 0
            elif best_score == gap_in_seq2:
                traceback[i, j] = 1
            else:
                traceback[i, j] = 2

    alignment_score = dp[L1, L2]
    aligned_indices = []
    i, j = L1, L2
    
    while i > 0 or j > 0:
        direction = traceback[i, j]
        if direction == 0:  
            aligned_indices.append((i-1, j-1))
            i -= 1
            j -= 1
        elif direction == 1:  
            aligned_indices.append((i-1, None))
            i -= 1
        else:  
            aligned_indices.append((None, j-1))
            j -= 1

    aligned_indices.reverse()
    return alignment_score, aligned_indices

# https://huggingface.co/docs/transformers/model_doc/esm
# https://huggingface.co/facebook/esm2_t36_3B_UR50D
def main():
    dataset_path="balibase_dataset/RV100"
    #fasta_file= extract_sequences(dataset_path)
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
    
    seq1 = str(seq_records[0].seq[:100])
    seq2 = str(seq_records[1].seq[:100])

    print(f"Sequence 1: {seq_records[0].id}, length={len(seq1)}")
    print(f"Sequence 2: {seq_records[1].id}, length={len(seq2)}")
    print(f"seq2: {seq1}")
    print(f"seq: {seq2}")
    
    seq1_embs = get_all_per_residue_embeddings(seq1, tokenizer, model,device)
    seq2_embs = get_all_per_residue_embeddings(seq2, tokenizer, model,device)
    alignment_score, alignment_indices = needleman_wunsch_align(seq1_embs, seq2_embs,distance_score,-4)

    aligned_seq1 = []
    aligned_seq2 = []
    for (i, j) in alignment_indices:
        if i is None:
            aligned_seq1.append("-")
        else:
            aligned_seq1.append(seq1[i])

        if j is None:
            aligned_seq2.append("-")
        else:
            aligned_seq2.append(seq2[j])

    aligned_seq1_str = "".join(aligned_seq1)
    aligned_seq2_str = "".join(aligned_seq2)
    print("Aligned Seq1: ", aligned_seq1_str)
    print("Aligned Seq2: ", aligned_seq2_str)
    print(f"Alignment score: {alignment_score}")
    
if __name__ == "__main__":
    main()
