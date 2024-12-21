# Weekly Progress

## Week 1:

- **Database Selection:**  
  Spent time researching various protein alignment benchmark datasets. Explored options like Pfam, SABmark, and OXBench, many of these databeses links where outdated or hard to find so in the end I settled on **BaliBASE** database.
- **Data Extraction & Initial Experimentation:**  
  Implemented a simple Python script to:
  1. Extract the protein sequences from the **BaliBASE** dataset.
  2. Process these sequences using **ESM2** to generate sequence embeddings.

## Week 2:

- **Implementation of Needleman–Wunsch for Sequence Alignment:**  
  Added a custom function, **needleman_wunsch_align**, to perform global sequence alignment on the ESM2 embeddings. This function takes per-residue embeddings for two sequences, applies a distance-based scoring scheme, and identifies the optimal alignment using the classic DP approach.  
  The current results look promisin and with a little fine tunning the algorithm will give a very good results and it looks like our aproch is going to work will for sequnce aligment.
