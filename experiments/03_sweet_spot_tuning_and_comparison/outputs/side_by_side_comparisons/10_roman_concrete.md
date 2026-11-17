# Example 10: Ancient Roman Concrete and Volcanic Ash Chemistry
**Domain**: Materials Science  
**Key Tracking Entities (17)**: `opus caementicium, Roman concrete, pozzolana, Pozzuoli, 2023, MIT, hot mixing, quicklime, lime clasts, calcium carbonate, self-healing, tobermorite, phillipsite, Pantheon, 43.3-meter, 125 AD, Hadrian`

### Original Source Text
> Ancient Roman concrete, known as opus caementicium, exhibits exceptional durability that has endured intact for over two millennia, surviving harsh maritime environments and seismic shocks. Roman builders combined slaked lime with volcanic ash known as pozzolana, harvested predominantly from volcanic deposits near Pozzuoli in the Gulf of Naples. Modern chemical analyses published in 2023 by researchers at MIT revealed that Romans utilized 'hot mixing' with quicklime (calcium oxide), forming millimetric lime clasts throughout the matrix. When cracks form and seawater infiltrates, these lime clasts dissolve and recrystallize as calcium carbonate, autogenously self-healing fractures. Furthermore, reaction with seawater promotes the growth of rare aluminous tobermorite and phillipsite crystals, strengthening the concrete over time. This ancient material allowed Emperor Hadrian to build the Pantheon's 43.3-meter unreinforced dome in 125 AD, which remains the largest in the world.

---

### Side-by-Side Generation Comparison

| Strategy              | Entity_Recall_%   |   Gen_PPL |   Distinct_1 |
|:----------------------|:------------------|----------:|-------------:|
| Top-1 (Greedy)        | 23.5%             |      1.52 |         0.76 |
| Standard Top-P (0.9)  | 17.6%             |      1.85 |         0.73 |
| Pure Post-Top-P (0.5) | 0.0%              |     22.57 |         0.74 |
| Tuned Hybrid          | 35.3%             |      1.06 |         0.86 |

#### [Top-1 (Greedy)] (Recall: 23.5% | PPL: 1.52 | Dist-1: 0.76)
```text
The ancient Romans, known for their innovative construction techniques, employed a combination of slaked lime and volcanic ash to create a durable, self-healing concrete that has endured for over two millennia. This remarkable material, known as opus caementicium, exhibits exceptional durability that has endured intact for over two millennia, surviving harsh maritime environments and seismic shocks.

The Romans utilized 'hot mixing' with quicklime (cal
```

#### [Standard Top-P (0.9)] (Recall: 17.6% | PPL: 1.85 | Dist-1: 0.73)
```text
The ancient Romans, known for their innovative construction techniques, employed a combination of slaked lime and volcanic ash to create a durable and resistant concrete. This combination of ingredients allowed them to build structures that were resistant to weathering, earthquakes, and sea-level changes. The Romans' use of hot mixing with quicklime (calcium oxide) and autogenously self-healing fractures enabled them to create strong, durable concrete that could
```

#### [Pure Post-Top-P (0.5)] (Recall: 0.0% | PPL: 22.57 | Dist-1: 0.74)
```text
To fully comprehend this historical account of Roman construction and construction technology in 1920, let us dive deeper. Roman construction, in this case study of a modern construction site in Naples (USA) provides an unparalleled perspective. Here, the archaeological context is vital. The construction process, in turn. dates to around the same period, around the time when Rome's strategic position at sea made its ships, including its largest
```

#### [Tuned Hybrid] (Recall: 35.3% | PPL: 1.06 | Dist-1: 0.86)
```text
Ancient Roman concrete, known as opus caementicium, exhibits exceptional durability that has endured intact for over two millennia, surviving harsh maritime environments and seismic shocks. Roman builders combined slaked lime with volcanic ash known as pozzolana, harvested predominantly from volcanic deposits near Pozzuoli in the Gulf of Naples. Modern chemical analyses published in 2023 by researchers at MIT revealed that Romans utilized 'hot
```

