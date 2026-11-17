# Example 02: CRISPR-Cas9 Bacterial Immunity to Precision Gene Editing
**Domain**: Biotechnology  
**Key Tracking Entities (17)**: `CRISPR, Cas9, Charpentier, Doudna, 2012, Science, single-guide RNA, sgRNA, PAM, Nobel Prize, 2020, 2023, FDA, Casgevy, Vertex, sickle cell, beta-thalassemia`

### Original Source Text
> CRISPR-Cas9 revolutionized molecular biology as a targeted genome editing platform. Discovered originally as an adaptive immune mechanism in bacteria against bacteriophages, CRISPR loci store spacer DNA sequences from past viral infections. In August 2012, researchers Emmanuelle Charpentier and Jennifer Doudna published their breakthrough paper in Science demonstrating that the Cas9 endonuclease could be programmed using a synthetic single-guide RNA (sgRNA) to cut double-stranded DNA at exact genomic coordinates adjacent to a protospacer adjacent motif (PAM). For this discovery, Charpentier and Doudna were awarded the Nobel Prize in Chemistry in 2020. In December 2023, regulatory agencies in the United Kingdom and the United States FDA approved Casgevy, the world's first CRISPR-based clinical gene therapy, developed by Vertex Pharmaceuticals and CRISPR Therapeutics to cure sickle cell disease and transfusion-dependent beta-thalassemia.

---

### Side-by-Side Generation Comparison

| Strategy              | Entity_Recall_%   |   Gen_PPL |   Distinct_1 |
|:----------------------|:------------------|----------:|-------------:|
| Top-1 (Greedy)        | 35.3%             |      1.13 |         0.89 |
| Standard Top-P (0.9)  | 35.3%             |      1.16 |         0.89 |
| Pure Post-Top-P (0.5) | 11.8%             |     28.24 |         0.69 |
| Tuned Hybrid          | 11.8%             |      2.46 |         0.68 |

#### [Top-1 (Greedy)] (Recall: 35.3% | PPL: 1.13 | Dist-1: 0.89)
```text
Here's a rewritten summary of the article:

CRISPR-Cas9 revolutionized molecular biology as a targeted genome editing platform. Discovered originally as an adaptive immune mechanism in bacteria against bacteriophages, CRISPR loci store spacer DNA sequences from past viral infections. In August 2012, researchers Emmanuelle Charpentier and Jennifer Doudna published their breakthrough paper in Science demonstrating that the Cas9 end
```

#### [Standard Top-P (0.9)] (Recall: 35.3% | PPL: 1.16 | Dist-1: 0.89)
```text
Here's a summary of the article:

CRISPR-Cas9 revolutionized molecular biology as a targeted genome editing platform. Discovered initially as an adaptive immune mechanism in bacteria against bacteriophages, CRISPR loci store spacer DNA sequences from past viral infections. In August 2012, researchers Emmanuelle Charpentier and Jennifer Doudna published their breakthrough paper in Science demonstrating that the Cas9 endon
```

#### [Pure Post-Top-P (0.5)] (Recall: 11.8% | PPL: 28.24 | Dist-1: 0.69)
```text
To write this summary:

2010
2.
Cisploride: a tool to measure gene regulation and disease diagnosis in clinical samples
CISPACE (CRISTS and Signature Gene-mediated Affected Pathway): CisPlorates is an emerging technology to quantify cis-Ploride binding in a sample of tissue.
2023-10
CRISPR Gene
```

#### [Tuned Hybrid] (Recall: 11.8% | PPL: 2.46 | Dist-1: 0.68)
```text
The breakthrough was a pivotal moment in the history of genetics, revolutionizing the field of molecular biology and enabling the development of personalized medicine.

The discovery of CRISPR-Cas9 revolutionized the field of molecular biology by introducing a targeted and efficient approach to gene editing. This innovative tool allowed for the modification of DNA sequences with unprecedented precision and accuracy, which has been instrumental in the development of many medical treatments and therapies.

In
```

