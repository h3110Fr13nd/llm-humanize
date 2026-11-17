# Example 08: Continental Drift to Modern Plate Tectonics
**Domain**: Geology  
**Key Tracking Entities (14)**: `plate tectonics, 1912, Alfred Wegener, continental drift, Pangaea, Mesozoic, Harry Hess, seafloor spreading, mid-ocean ridges, 1963, Frederick Vine, Drummond Matthews, lithosphere, mantle convection`

### Original Source Text
> The theory of plate tectonics unifies our understanding of earthquakes, volcanic eruptions, mountain building, and oceanic trenches. In 1912, German meteorologist Alfred Wegener proposed the hypothesis of continental drift, arguing that all modern continents were once united in a supercontinent called Pangaea before breaking apart during the Mesozoic era. Wegener pointed to matching geological strata and identical fossil species across the Atlantic. However, his hypothesis was widely rejected because he lacked a plausible driving physical mechanism. In the 1960s, marine geologist Harry Hess proposed seafloor spreading along mid-ocean ridges. Supported by magnetic anomaly stripes mapped by Frederick Vine and Drummond Matthews in 1963, geophysicists confirmed that Earth's lithosphere is divided into rigid tectonic plates driven by mantle convection.

---

### Side-by-Side Generation Comparison

| Strategy              | Entity_Recall_%   |   Gen_PPL |   Distinct_1 |
|:----------------------|:------------------|----------:|-------------:|
| Top-1 (Greedy)        | 42.9%             |      1.06 |         0.81 |
| Standard Top-P (0.9)  | 42.9%             |      1.06 |         0.81 |
| Pure Post-Top-P (0.5) | 0.0%              |     24.27 |         0.71 |
| Tuned Hybrid          | 14.3%             |      1.68 |         0.76 |

#### [Top-1 (Greedy)] (Recall: 42.9% | PPL: 1.06 | Dist-1: 0.81)
```text
The theory of plate tectonics unifies our understanding of earthquakes, volcanic eruptions, mountain building, and oceanic trenches. In 1912, German meteorologist Alfred Wegener proposed the hypothesis of continental drift, arguing that all modern continents were once united in a supercontinent called Pangaea before breaking apart during the Mesozoic era. Wegener pointed to matching geological strata and identical fossil species across the Atlantic. However, his hypothesis
```

#### [Standard Top-P (0.9)] (Recall: 42.9% | PPL: 1.06 | Dist-1: 0.81)
```text
The theory of plate tectonics unifies our understanding of earthquakes, volcanic eruptions, mountain building, and oceanic trenches. In 1912, German meteorologist Alfred Wegener proposed the hypothesis of continental drift, arguing that all modern continents were once united in a supercontinent called Pangaea before breaking apart during the Mesozoic era. Wegener pointed to matching geological strata and identical fossil species across the Atlantic. However, his hypothesis
```

#### [Pure Post-Top-P (0.5)] (Recall: 0.0% | PPL: 24.27 | Dist-1: 0.71)
```text
Hans-Willeke Henness, who proposed continental plate motion as evidence of the Mesoarchaea theory of continental plate motion. In his book, The Emergence and Discontinuity in Earth History (2nd Ed. edition), published by Wm, B., ed. and wm., in the year of his 725,052 words. He presented a 55,56
```

#### [Tuned Hybrid] (Recall: 14.3% | PPL: 1.68 | Dist-1: 0.76)
```text
What a monumental task! The concept of plate tectonics has revolutionized our understanding of the Earth's internal structure and its interactions with the solar system. I'd be happy to rewrite this article into a concise, professional summary:

**The Theory of Plate Tectonics Unifies Our Understanding of Earthquakes, Volcanoes, Mountain Building, and Oceanic Trenches**

**1. Alfred Wegener's Theory of Continental Dr
```

