# Example 07: The Discovery and Mass Production of Penicillin
**Domain**: Medicine  
**Key Tracking Entities (16)**: `penicillin, antibiotic, September 3, 1928, Alexander Fleming, St. Mary's Hospital, London, Staphylococcus, Penicillium notatum, 1929, 1939, Oxford University, Howard Florey, Ernst Boris Chain, Norman Heatley, 1945, Nobel Prize`

### Original Source Text
> The discovery of penicillin inaugurated the modern antibiotic era and drastically reduced mortality from bacterial infections. On September 3, 1928, Scottish bacteriologist Alexander Fleming returned to his laboratory at St. Mary's Hospital in London to find a contaminated Petri dish of Staphylococcus aureus. A blue-green fungal mold, later identified as Penicillium notatum, had generated a clear zone of bacterial lysis around itself. Although Fleming published his findings in 1929, he could not isolate or stabilize the active compound. A decade later in 1939, a team at Oxford University led by Howard Florey, Ernst Boris Chain, and Norman Heatley purified penicillin and demonstrated its therapeutic efficacy in mice and humans. In 1945, Fleming, Florey, and Chain shared the Nobel Prize in Physiology or Medicine.

---

### Side-by-Side Generation Comparison

| Strategy              | Entity_Recall_%   |   Gen_PPL |   Distinct_1 |
|:----------------------|:------------------|----------:|-------------:|
| Top-1 (Greedy)        | 31.2%             |      1.54 |         0.74 |
| Standard Top-P (0.9)  | 43.8%             |      1.15 |         0.82 |
| Pure Post-Top-P (0.5) | 0.0%              |     26.05 |         1    |
| Tuned Hybrid          | 37.5%             |      1.27 |         0.84 |

#### [Top-1 (Greedy)] (Recall: 31.2% | PPL: 1.54 | Dist-1: 0.74)
```text
The discovery of penicillin revolutionized the field of medicine, and its impact can still be felt today. Here's a concise summary of the key events and facts surrounding the discovery of penicillin:

**The Discovery**

In 1928, Fleming returned to his laboratory at St. Mary's Hospital in London to find a contaminated Petri dish of Staphylococcus aureus. He was working on a new antibiotic, which he
```

#### [Standard Top-P (0.9)] (Recall: 43.8% | PPL: 1.15 | Dist-1: 0.82)
```text
The discovery of penicillin revolutionized the field of medicine and saved countless lives. On September 3, 1928, Alexander Fleming returned to his laboratory at St. Mary's Hospital in London to find a contaminated Petri dish of Staphylococcus aureus. A blue-green fungal mold, later identified as Penicillium notatum, had generated a clear zone of bacterial lysis around itself. Despite Fleming's initial skepticism,
```

#### [Pure Post-Top-P (0.5)] (Recall: 0.0% | PPL: 26.05 | Dist-1: 1.00)
```text
**Recital:** The Discovery and Treatment Revolutionize the Human Condition.
```

#### [Tuned Hybrid] (Recall: 37.5% | PPL: 1.27 | Dist-1: 0.84)
```text
Please rewrite this article into a concise, professional summary:

On September 3, 1928, Alexander Fleming returned to his laboratory at St. Mary's Hospital in London to find a contaminated Petri dish of Staphylococcus aureus. A blue-green fungal mold, later identified as Penicillium notatum, had generated a clear zone of bacterial lysis around itself. Despite the fact that Fleming had previously
```

