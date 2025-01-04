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

################################################
### negative L2 distance based similarity
################################################
def distance_score(emb1, emb2):
    dist = torch.norm(emb1 - emb2, p=2)
    return -dist.item()

###############################################################################
### raw distance funtion for the original sequences (not embedding )
###############################################################################
def raw_distance_score(char1, char2):
    return 1 if char1 == char2 else -1

###################################################
### needleman-wunsch but for normal sequences
####################################################
def needleman_wunsch_align_raw(seq1, seq2, scoring_func, gap_penalty=-5.0):
    L1 = len(seq1)
    L2 = len(seq2)
    dp = np.zeros((L1 + 1, L2 + 1), dtype=np.float32)
    traceback = np.zeros((L1 + 1, L2 + 1), dtype=np.int32)

    for i in range(1, L1 + 1):
        dp[i, 0] = dp[i - 1, 0] + gap_penalty
        traceback[i, 0] = 1
    for j in range(1, L2 + 1):
        dp[0, j] = dp[0, j - 1] + gap_penalty
        traceback[0, j] = 2

    for i in range(1, L1 + 1):
        for j in range(1, L2 + 1):
            match_score = dp[i - 1, j - 1] + scoring_func(seq1[i - 1], seq2[j - 1])
            gap_in_seq2 = dp[i - 1, j] + gap_penalty
            gap_in_seq1 = dp[i, j - 1] + gap_penalty

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
            aligned_indices.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif direction == 1:
            aligned_indices.append((i - 1, None))
            i -= 1
        else:
            aligned_indices.append((None, j - 1))
            j -= 1

    aligned_indices.reverse()
    return alignment_score, aligned_indices
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


def count_matches(aligned_s1: str, aligned_s2: str) -> int:
    matches = 0
    for a, b in zip(aligned_s1, aligned_s2):
        if a != '-' and b != '-' and a == b:
            matches += 1
    return matches

def reconstruct_alignment(alignment, s1, s2):
    aligned_s1 = []
    aligned_s2 = []
    prev_end1 = 0
    prev_end2 = 0

    for (start1, end1), (start2, end2) in zip(alignment.aligned[0], alignment.aligned[1]):
        
        if start1 > prev_end1:
            gap_length_1 = start1 - prev_end1
            aligned_s1.append('-' * gap_length_1)
            aligned_s2.append(s2[prev_end2 : prev_end2 + gap_length_1])
            prev_end2 += gap_length_1

        if start2 > prev_end2:
            gap_length_2 = start2 - prev_end2
            aligned_s2.append('-' * gap_length_2)
            aligned_s1.append(s1[prev_end1 : prev_end1 + gap_length_2])
            prev_end1 += gap_length_2

        aligned_s1.append(s1[start1:end1])
        aligned_s2.append(s2[start2:end2])
        prev_end1 = end1
        prev_end2 = end2

    final_s1 = "".join(aligned_s1)
    final_s2 = "".join(aligned_s2)
    return final_s1, final_s2

# https://huggingface.co/docs/transformers/model_doc/esm
# https://huggingface.co/facebook/esm2_t36_3B_UR50D
def main():
    dataset_path = "balibase_dataset/RV100"
    extract_sequences(dataset_path)
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
    
    seq1 = str(seq_records[2].seq)
    seq2 = str(seq_records[1].seq)
    
    # Embedding-based alignment
    seq1_embs = get_all_per_residue_embeddings(seq1, tokenizer, model, device)
    seq2_embs = get_all_per_residue_embeddings(seq2, tokenizer, model, device)
    alignment_score_emb, alignment_indices_emb = needleman_wunsch_align(
        seq1_embs, seq2_embs, distance_score, gap_penalty=-5
    )
    aligned_seq1_emb = "".join(seq1[i] if i is not None else "-" for i, j in alignment_indices_emb)
    aligned_seq2_emb = "".join(seq2[j] if j is not None else "-" for i, j in alignment_indices_emb)
    matches_emb = count_matches(aligned_seq1_emb, aligned_seq2_emb)

    # Raw sequence alignment
    alignment_score_raw, alignment_indices_raw = needleman_wunsch_align_raw(
        seq1, seq2, raw_distance_score, gap_penalty=-5
    )
    aligned_seq1_raw = "".join(seq1[i] if i is not None else "-" for i, j in alignment_indices_raw)
    aligned_seq2_raw = "".join(seq2[j] if j is not None else "-" for i, j in alignment_indices_raw)
    matches_raw = count_matches(aligned_seq1_raw, aligned_seq2_raw)

    # BioPython's PairwiseAligner
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    alignments = aligner.align(seq1, seq2)
    best_alignment = alignments[0]
    bp_aligned_s1, bp_aligned_s2 = reconstruct_alignment(best_alignment, seq1, seq2)
    bp_matches = count_matches(bp_aligned_s1, bp_aligned_s2)
    
    print(f"Embedding-based alignment matches: {matches_emb}")
    print(f"Raw sequence alignment matches: {matches_raw}")
    print(f"BioPython's PairwiseAligner number of matches: {bp_matches}")
if __name__ == "__main__":
    main()