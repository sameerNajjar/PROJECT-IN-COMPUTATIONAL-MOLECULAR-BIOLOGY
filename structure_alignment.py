from huggingface_hub import login
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, ProteinChain, LogitsConfig
from Bio.PDB import PDBParser
import torch
import numpy as np
import os
from dotenv import load_dotenv

######################################################
### get the ID of the protien chains from pdb file
######################################################
def get_chain_ids(pdb_path: str):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    chain_ids = [chain.id for chain in structure.get_chains()]
    return chain_ids

###########################################
### Cosine similarity-based scoring
###########################################
def cosine_distance_score(emb1, emb2):
    cosine_similarity = torch.dot(emb1, emb2) / (torch.norm(emb1) * torch.norm(emb2))
    return cosine_similarity.item()

###########################################
### L2 distance scoring
###########################################
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

def main():
    load_dotenv(dotenv_path="environment.env")
    token = os.getenv("HF_TOKEN")
    login(token)  

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model: ESM3InferenceClient = ESM3.from_pretrained("esm3-open").to(device).eval()

    # Load PDB files and extract chain IDs
    pdb1_path = "extracted_pdb_files/AF-Q0P967-F1-model_v4.pdb"
    pdb2_path = "extracted_pdb_files/AF-O52908-F1-model_v4.pdb"

    chain_ID1 = get_chain_ids(pdb1_path)
    chain_ID2 = get_chain_ids(pdb2_path)

    # Load proteins and convert to ESMProtein
    protein_chain1 = ProteinChain.from_pdb(pdb1_path, chain_id=chain_ID1[0])
    protein1 = ESMProtein.from_protein_chain(protein_chain1)

    protein_chain2 = ProteinChain.from_pdb(pdb2_path, chain_id=chain_ID2[0])
    protein2 = ESMProtein.from_protein_chain(protein_chain2)

    # Encode proteins and extract embeddings
    protein_tensor1 = model.encode(protein1)
    logits_output1 = model.logits(
        protein_tensor1,
        LogitsConfig(return_embeddings=True, structure=True)
    )
    embds1 = logits_output1.embeddings.squeeze(0).to(device)  # Move to device

    protein_tensor2 = model.encode(protein2)
    logits_output2 = model.logits(
        protein_tensor2,
        LogitsConfig(return_embeddings=True, structure=True)
    )
    embds2 = logits_output2.embeddings.squeeze(0).to(device)  # Move to device

    # Perform alignment
    score, indices = needleman_wunsch_align(embds1, embds2, cosine_distance_score)

if __name__ == "__main__":
    main()