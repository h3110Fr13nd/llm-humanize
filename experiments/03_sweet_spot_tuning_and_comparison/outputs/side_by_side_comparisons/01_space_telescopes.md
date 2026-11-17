# Example 01: The Evolution of Space Telescopes: Hubble and JWST
**Domain**: Astronomy  
**Key Tracking Entities (22)**: `Hubble, 1990, 540, 2.4-meter, COSTAR, 1993, 1995, Deep Field, 1998, dark energy, JWST, 6.5-meter, beryllium, L2, 1.5 million, Kapton, 50 Kelvin, MIRI, 7 Kelvin, December 25, 2021, Ariane 5, Kourou`

### Original Source Text
> The development of astronomical space telescopes represents one of humanity's greatest scientific achievements. The Hubble Space Telescope was deployed into Low Earth Orbit at an altitude of approximately 540 kilometers by Space Shuttle Discovery on April 24, 1990. Hubble featured a 2.4-meter primary mirror. Initially, a spherical aberration of 2.2 micrometers prevented sharp focus until NASA astronauts installed COSTAR in December 1993 during STS-61. In 1995, Hubble captured the Deep Field image revealing over 3,000 ancient galaxies. Later, observations of Type Ia supernovae in 1998 proved the universe's expansion is accelerating due to dark energy. In 2021, NASA launched the James Webb Space Telescope (JWST) aboard an Ariane 5 rocket from Kourou on December 25, 2021. Positioned at Lagrange Point 2 (L2), 1.5 million kilometers from Earth, JWST features a 6.5-meter gold-coated beryllium mirror and a five-layer Kapton sunshield cooling instruments below 50 Kelvin. Its MIRI instrument is actively refrigerated to 7 Kelvin.

---

### Side-by-Side Generation Comparison

| Strategy              | Entity_Recall_%   |   Gen_PPL |   Distinct_1 |
|:----------------------|:------------------|----------:|-------------:|
| Top-1 (Greedy)        | 9.1%              |      1.64 |         0.69 |
| Standard Top-P (0.9)  | 9.1%              |      1.71 |         0.69 |
| Pure Post-Top-P (0.5) | 0.0%              |     13.13 |         0.9  |
| Tuned Hybrid          | 9.1%              |      2.22 |         0.55 |

#### [Top-1 (Greedy)] (Recall: 9.1% | PPL: 1.64 | Dist-1: 0.69)
```text
Here's a summary of the article's key points:

**Development of Astronomical Space Telescopes:**

* Hubble Space Telescope (HST) was launched in 1990 and has been operational since 2009.
* Hubble is a space-based observatory that has captured stunning images of the universe, including the cosmic microwave background radiation, the cosmic web, and the formation of galaxies
```

#### [Standard Top-P (0.9)] (Recall: 9.1% | PPL: 1.71 | Dist-1: 0.69)
```text
Here's a summary of the article:

**The Development of Astronomical Space Telescopes:**

The Hubble Space Telescope (HST) was launched in 1990 and has been operating continuously since 2009. The telescope is a space-based observatory that has been providing high-resolution images of the universe since its launch. Hubble has been observing the universe for over 25 years
```

#### [Pure Post-Top-P (0.5)] (Recall: 0.0% | PPL: 13.13 | Dist-1: 0.90)
```text
What an excellent article. Here is the full summary, please read the first 15 lines.
```

#### [Tuned Hybrid] (Recall: 9.1% | PPL: 2.22 | Dist-1: 0.55)
```text
To write a concise, professional summary of the development of the Hubble Space Telescope, I'll use the following approach:

I. Briefly introduce the history of the Hubble Space Telescope and its importance.

II. Discuss the scientific achievements of the Hubble Space Telescope, including its groundbreaking discoveries and discoveries made in the late 1990s.

III. Explain the significance of the development of the Hubble
```

