# Example 04: Johannes Gutenberg and the Printing Revolution
**Domain**: History  
**Key Tracking Entities (13)**: `Johannes Gutenberg, Mainz, 1440, movable type, alloy, oil-based ink, 1455, 42-Line, Gutenberg Bible, Renaissance, Martin Luther, 1517, Scientific Revolution`

### Original Source Text
> The invention of the movable metal type mechanical printing press by Johannes Gutenberg in Mainz, Germany, around 1440 transformed European civilization. Prior to Gutenberg, books were laboriously transcribed by hand by scribes or carved into woodblocks, making written knowledge prohibitively scarce and expensive. Gutenberg combined movable type cast from a lead-tin-antimony alloy with oil-based ink and an adapted wooden screw agricultural press. In 1455, Gutenberg completed his masterwork, the 42-Line Gutenberg Bible, printing approximately 180 copies on paper and vellum. The rapid spread of printing presses across Europe catalyzed the Renaissance, facilitated Martin Luther's Protestant Reformation in 1517, and accelerated the Scientific Revolution by allowing rapid, accurate dissemination of empirical discoveries.

---

### Side-by-Side Generation Comparison

| Strategy              | Entity_Recall_%   |   Gen_PPL |   Distinct_1 |
|:----------------------|:------------------|----------:|-------------:|
| Top-1 (Greedy)        | 38.5%             |      1.21 |         0.79 |
| Standard Top-P (0.9)  | 38.5%             |      1.21 |         0.79 |
| Pure Post-Top-P (0.5) | 0.0%              |     13.75 |         0.81 |
| Tuned Hybrid          | 23.1%             |      1.72 |         0.78 |

#### [Top-1 (Greedy)] (Recall: 38.5% | PPL: 1.21 | Dist-1: 0.79)
```text
Here's a rewritten summary of the article:

The invention of the movable metal type mechanical printing press by Johannes Gutenberg in Mainz, Germany, around 1440 revolutionized the way knowledge was produced and disseminated. Prior to Gutenberg, books were laboriously transcribed by scribes or carved into woodblocks, making written knowledge prohibitively scarce and expensive. Gutenberg combined movable type cast from a lead-tin-antimony alloy
```

#### [Standard Top-P (0.9)] (Recall: 38.5% | PPL: 1.21 | Dist-1: 0.79)
```text
Here's a rewritten summary of the article:

The invention of the movable metal type mechanical printing press by Johannes Gutenberg in Mainz, Germany, around 1440 revolutionized the way knowledge was produced and disseminated. Prior to Gutenberg, books were laboriously transcribed by scribes or carved into woodblocks, making written knowledge prohibitively scarce and expensive. Gutenberg combined movable type cast from a lead-tin-antimony alloy
```

#### [Pure Post-Top-P (0.5)] (Recall: 0.0% | PPL: 13.75 | Dist-1: 0.81)
```text
To write this summary accurately and effectively:

 I would break the text up in 3 paragraphs. I'd begin by describing what Gutenberg was, his inventions and their impact, then transition to summarizing his main arguments, including key historical events, technological innovations and innovations that have influenced society, then end by concluding that he successfully applied these ideas and made significant changes in Europe's economy.
```

#### [Tuned Hybrid] (Recall: 23.1% | PPL: 1.72 | Dist-1: 0.78)
```text
To rewrite this article into a concise, professional summary:

The invention of the movable metal type mechanical printing press by Johannes Gutenberg in Mainz, Germany, around 1440 revolutionized the spread of knowledge and cultural transmission by transforming the physical record of texts from manuscript to paper and printing. Prior to Gutenberg, books were laboriously transcribed by scribes or carved into woodblocks, making written knowledge prohibitively scarce and expensive
```

