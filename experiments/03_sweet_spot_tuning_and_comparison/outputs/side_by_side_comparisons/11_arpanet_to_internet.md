# Example 11: ARPANET to the Global Internet Protocol
**Domain**: Computer Networking  
**Key Tracking Entities (21)**: `ARPANET, DARPA, 1969, October 29, 1969, UCLA, SRI, 1974, Vinton Cerf, Robert Kahn, TCP, IP, TCP/IP, January 1, 1983, flag day, NCP, Tim Berners-Lee, World Wide Web, CERN, 1989, HTML, HTTP`

### Original Source Text
> The global internet evolved from military research into decentralized packet-switched communication networks during the Cold War. In 1969, the United States Department of Defense's Advanced Research Projects Agency (DARPA) established ARPANET. The first successful message was transmitted on October 29, 1969, between Leonard Kleinrock's laboratory at UCLA and Douglas Engelbart's SRI International system. In 1974, computer scientists Vinton Cerf and Robert Kahn published their seminal paper designing the Transmission Control Protocol (TCP), which was subsequently split into TCP and Internet Protocol (IP). On January 1, 1983, known as 'flag day,' ARPANET officially switched from the original NCP protocol to the TCP/IP protocol suite, creating the modern Internet. Tim Berners-Lee later developed the World Wide Web at CERN in 1989, introducing HTML, HTTP, and URLs.

---

### Side-by-Side Generation Comparison

| Strategy              | Entity_Recall_%   |   Gen_PPL |   Distinct_1 |
|:----------------------|:------------------|----------:|-------------:|
| Top-1 (Greedy)        | 19.0%             |      1.16 |         0.78 |
| Standard Top-P (0.9)  | 9.5%              |      1.28 |         0.79 |
| Pure Post-Top-P (0.5) | 0.0%              |     25.31 |         0.73 |
| Tuned Hybrid          | 4.8%              |      1.86 |         0.62 |

#### [Top-1 (Greedy)] (Recall: 19.0% | PPL: 1.16 | Dist-1: 0.78)
```text
Here's a rewritten summary of the article:

The global internet evolved from military research into decentralized packet-switched communication networks during the Cold War. In 1969, the United States Department of Defense's Advanced Research Projects Agency (DARPA) established ARPANET, a precursor to the modern Internet. The first successful message was transmitted on October 29, 1969, between Leonard
```

#### [Standard Top-P (0.9)] (Recall: 9.5% | PPL: 1.28 | Dist-1: 0.79)
```text
Here's a rewritten summary of the article:

**The Global Internet Evolution from Military Research to Decentralized Packet-Switched Communication Networks**

The global internet, which we now call the Internet, evolved from military research into decentralized packet-switched communication networks during the Cold War. In 1969, the United States Department of Defense's Advanced Research Projects Agency (DARPA) established ARP
```

#### [Pure Post-Top-P (0.5)] (Recall: 0.0% | PPL: 25.31 | Dist-1: 0.73)
```text
To write this summary:

2004: "1.25 years of experimentation with packet-based network design and architecture.
1.125 year's research in ARPanatIC (arpp). 2.05 year of effort to merge packet and protocol stack."
```

#### [Tuned Hybrid] (Recall: 4.8% | PPL: 1.86 | Dist-1: 0.62)
```text
To rewrite this article into a concise, professional summary, I'd use the following structure:

I. Introduction

* Briefly introduce the topic of the article and its significance
* Provide context about the internet's evolution

II. History of the Internet

* Describe the origins of the internet and its evolution from military research to decentralized network
* Explain the concept of the Internet Protocol (IP) and
```

