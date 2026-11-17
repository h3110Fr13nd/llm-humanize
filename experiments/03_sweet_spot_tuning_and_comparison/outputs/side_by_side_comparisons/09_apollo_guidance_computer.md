# Example 09: The Apollo Guidance Computer and Real-Time Software
**Domain**: Aerospace  
**Key Tracking Entities (16)**: `Apollo Guidance Computer, AGC, Apollo 11, MIT, Eldon Hall, Margaret Hamilton, integrated circuits, NOR gates, 1.024 megahertz, core rope memory, July 20, 1969, 1201, 1202, priority scheduling, Neil Armstrong, Buzz Aldrin`

### Original Source Text
> The Apollo Guidance Computer (AGC) was a pioneering achievement in computer engineering and software architecture that enabled the Apollo 11 lunar landing. Designed at the MIT Instrumentation Laboratory under the leadership of Eldon Hall and software director Margaret Hamilton, the AGC was among the first computers to utilize silicon integrated circuits (ICs), employing approximately 5,600 NOR gates. Operating with a clock speed of 1.024 megahertz, the AGC had 2,048 words of RAM and 36,864 words of core rope memory woven by hand. During Apollo 11's descent on July 20, 1969, an overloaded radar system triggered famous 1201 and 1202 program alarms. Margaret Hamilton's asynchronous priority scheduling design allowed the AGC to drop low-priority radar tasks while preserving vital lunar thruster control routines, ensuring Neil Armstrong and Buzz Aldrin landed safely.

---

### Side-by-Side Generation Comparison

| Strategy              | Entity_Recall_%   |   Gen_PPL |   Distinct_1 |
|:----------------------|:------------------|----------:|-------------:|
| Top-1 (Greedy)        | 37.5%             |      1.25 |         0.65 |
| Standard Top-P (0.9)  | 50.0%             |      1.1  |         0.75 |
| Pure Post-Top-P (0.5) | 0.0%              |     20.29 |         0.75 |
| Tuned Hybrid          | 12.5%             |      2.18 |         0.71 |

#### [Top-1 (Greedy)] (Recall: 37.5% | PPL: 1.25 | Dist-1: 0.65)
```text
Here's a rewritten summary of the Apollo Guidance Computer (AGC) article:

**The Apollo Guidance Computer (AGC) - A Pioneering Achievement**

The Apollo Guidance Computer (AGC) was a pioneering achievement in computer engineering and software architecture that enabled the Apollo 11 lunar landing. Designed at the MIT Instrumentation Laboratory under the leadership of Eldon Hall and software director Margaret Hamilton, the A
```

#### [Standard Top-P (0.9)] (Recall: 50.0% | PPL: 1.10 | Dist-1: 0.75)
```text
The Apollo Guidance Computer (AGC) was a pioneering achievement in computer engineering and software architecture that enabled the Apollo 11 lunar landing. Designed at the MIT Instrumentation Laboratory under the leadership of Eldon Hall and software director Margaret Hamilton, the AGC was among the first computers to utilize silicon integrated circuits (ICs) and employed approximately 5,600 NOR gates. Operating with a clock speed of 1.
```

#### [Pure Post-Top-P (0.5)] (Recall: 0.0% | PPL: 20.29 | Dist-1: 0.75)
```text
To write this summary accurately and effectively:

 I would break it into sections and create sub-submissions based upon these subsections:

 I. History: Brief overview, timeline of Apollo's accomplishments

 II: Technology and Architecture
 I would present an in depth summary, using key technological terms, like CPU architecture (intellectual architecture) or AGI architecture. This section should cover:
	+ 5-
```

#### [Tuned Hybrid] (Recall: 12.5% | PPL: 2.18 | Dist-1: 0.71)
```text
To rewrite this article into a concise, professional summary, I'd suggest the following steps:

1. Identify the key historical facts and dates that need to be included:
	* The Apollo 11 landing was achieved by 10:17 PM EST on July 20, 1969.
	* The landing marks the first time a spacecraft has touched down on the moon's surface
```

