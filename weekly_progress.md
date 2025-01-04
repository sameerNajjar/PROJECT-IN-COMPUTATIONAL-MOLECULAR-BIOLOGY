# Weekly Progress

## Week 1:

- **Database Selection:**  
  Spent time researching various protein alignment benchmark datasets. Explored options like Pfam, SABmark, and OXBench, many of these databeses links where outdated or hard to find so in the end I settled on **BaliBASE** database.

- **Database Download Link:**

  - [BaliBASE Download](http://www.lbgi.fr/balibase/BalibaseDownload/)

- **Data Extraction & Initial Experimentation:**  
  Implemented a simple Python script to:
  1. Extract the protein sequences from the **BaliBASE** dataset.
  2. Process these sequences using **ESM2** to generate sequence embeddings.

## Week 2:

- **Implementation of Needleman–Wunsch for Sequence Alignment:**
  - Designed and implemented the `needleman_wunsch_align` function for global sequence alignment.
  - Utilized **ESM2 embeddings** to represent protein sequences and applied a distance-based scoring scheme.
  - Achieved promising initial results, suggesting that the approach is effective for sequence alignment. Further fine-tuning and testing is planned to improve accuracy and performance.

## Week 3:

- **Enhancements to the Algorithm:**

  - Experimented with various modifications to improve alignment results:
    - Tested different gap penalties.
    - Evaluated multiple distance functions, including:
      - **Negative L2 Distance**
      - **Cosine Similarity**
    - Explored alternative alignment algorithms such as:
      - **Smith-Waterman**
      - **Dynamic Time Warping**
      - **Progressive Alignment**
  - got the best results with the **Needleman–Wunsch algorithm** combined with **Negative L2 Distance** or **Cosine Similarity**.

- **Comparison with BLAST:**

  - Compared the algorithm’s performance against **BLAST** using the [NCBI BLAST website](https://blast.ncbi.nlm.nih.gov/Blast.cgi).
  - Results showed that the my algorithm produced better results on must tested sequences.

- **Embedding vs. original Sequence Comparison:**
  - Compared the `needleman_wunsch_align` algorithm on **ESM2 embeddings** versus **original raw sequences**:
    - Observed slightly better results with raw sequences.
    - I think that the use of **ESM2 (150M parameters)** might limit the embedding’s expressiveness.
    - Plan to upgrade to **ESM3** next week for improved embedding quality.
