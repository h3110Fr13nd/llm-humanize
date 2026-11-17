# Example 06: The Voyager Interstellar Mission and Golden Record
**Domain**: Space Exploration  
**Key Tracking Entities (18)**: `Voyager, 1977, Cape Canaveral, Voyager 2, August 20, 1977, Voyager 1, September 5, 1977, 175 years, Jupiter, Saturn, Uranus, Neptune, Golden Record, Carl Sagan, 2012, heliopause, interstellar space, 2018`

### Original Source Text
> NASA's Voyager program remains one of the most audacious deep-space exploration ventures in human history. Launched in late summer 1977 from Cape Canaveral, Voyager 2 lifted off on August 20, 1977, followed by Voyager 1 on September 5, 1977, taking advantage of a rare planetary alignment occurring once every 175 years to tour Jupiter, Saturn, Uranus, and Neptune. Both spacecraft carry a 12-inch gold-plated copper phonograph record curated by Carl Sagan, containing 115 images, sounds of Earth, musical selections from Bach to Chuck Berry, and spoken greetings in 55 languages. In August 2012, Voyager 1 crossed the heliopause at 121 astronomical units (AU) from the Sun, becoming the first human-made object to enter interstellar space. Voyager 2 crossed into interstellar space in November 2018.

---

### Side-by-Side Generation Comparison

| Strategy              | Entity_Recall_%   |   Gen_PPL |   Distinct_1 |
|:----------------------|:------------------|----------:|-------------:|
| Top-1 (Greedy)        | 11.1%             |      1.73 |         0.64 |
| Standard Top-P (0.9)  | 11.1%             |      1.73 |         0.64 |
| Pure Post-Top-P (0.5) | 0.0%              |     41.24 |         0.9  |
| Tuned Hybrid          | 38.9%             |      1.1  |         0.64 |

#### [Top-1 (Greedy)] (Recall: 11.1% | PPL: 1.73 | Dist-1: 0.64)
```text
Here's a summary of the article:

**The Voyager Program**

The Voyager program, launched in 1977, was a groundbreaking mission that aimed to explore the outer planets and beyond. The spacecraft, which consisted of six scientific instruments, was designed to study the atmospheres, magnetic fields, and potential biosignatures of the outer planets. The program was a collaborative effort between the United States, the
```

#### [Standard Top-P (0.9)] (Recall: 11.1% | PPL: 1.73 | Dist-1: 0.64)
```text
Here's a summary of the article:

**The Voyager Program**

The Voyager program, launched in 1977, was a groundbreaking mission that aimed to explore the outer planets and beyond. The spacecraft, which consisted of six scientific instruments, was designed to study the atmospheres, magnetic fields, and potential biosignatures of the outer planets. The program was a collaborative effort between the United States, the
```

#### [Pure Post-Top-P (0.5)] (Recall: 0.0% | PPL: 41.24 | Dist-1: 0.90)
```text
**Title: The Journey to Intergalaxies**

 Subheading
A Global Voynium of a Year, in One Brief Sentence!

 Verification

 Impostor Alert: You are the original poster, your authenticity verified.
```

#### [Tuned Hybrid] (Recall: 38.9% | PPL: 1.10 | Dist-1: 0.64)
```text
NASA's Voyager program remains one of the most audacious deep-space exploration ventures in human history. Launched in late summer 1977 from Cape Canaveral, Florida, Voyager 2 lifted off on August 20, 1977, followed by Voyager 1 on September 5, 1977, taking advantage of a rare planetary alignment occurring once every 175
```

