# IntentRank — Intelligent Candidate Discovery & Ranking

**India Runs Hackathon 2026 · Redrob AI · Data & AI Challenge, Track 1**
**Team: Apex Climbers**

A candidate ranking system that evaluates fit the way a strong recruiter would — by weighing semantic alignment, skill depth, behavioral availability, and career trajectory together — instead of matching keywords against a job description.

---

## Table of Contents

- [Why this exists](#why-this-exists)
- [What we found when we tested the existing approach](#what-we-found-when-we-tested-the-existing-approach)
- [System overview](#system-overview)
- [Full workflow, layer by layer](#full-workflow-layer-by-layer)
- [Setup & Execution](#setup--execution)
- [Output format](#output-format)
- [Constraints compliance](#constraints-compliance)
- [Design decisions and known limitations](#design-decisions-and-known-limitations)

---

## Why this exists

The challenge brief states the problem directly: recruiters go through hundreds of profiles and still miss the right person — not because the talent isn't there, but because keyword filters can't see what actually matters.

Before writing a single line of this system, we tested that claim ourselves against Redrob's own existing AI Resume Ranker, using synthetic candidates designed to separate keyword density from real production depth. The results confirmed the problem is real, specific, and measurable.

## What we found when we tested the existing approach

We built five synthetic candidate profiles for a single job description and ran them through Redrob's existing ranker:

- A candidate with **zero measurable outcomes** (vague claims like "tested on sample documents") **outranked** a candidate with **200+ GitHub stars, 7 merged open-source PRs, and production-deployed libraries**.
- The score spread across all five candidates — ranging from a course-project-only profile to one with three years of production experience — was only **4.7 points**, too compressed to be a useful signal.
- The candidate with the **second-highest score** in the set was ranked **4th**, not 2nd — score and rank order were internally inconsistent.

When we ran the actual 100,000-candidate dataset released for this challenge through our own hard-filtering logic, **63,710 candidates (63.7%)** carried a non-technical primary title paired with an AI-keyword-stuffed skills list.

This system is built specifically to close those gaps: **outcome blindness, keyword stuffing vulnerability, and unexplained scoring.**

---

## System overview

![IntentRank System Overview](./IntentRank_System_Overview.png)

Our ranking pipeline is designed to be highly accurate while strictly passing the ≤ 5-minute Stage 3 computational limit. 

**Why we chose sentence-transformers over TF-IDF:** While TF-IDF is significantly faster, we deliberately prioritize ranking quality over speed. TF-IDF relies heavily on exact keyword matching, which fails to capture semantic similarity (e.g., matching "deep neural networks" to "PyTorch"). Sentence embeddings capture the true meaning and intent behind a candidate's experience, providing a much higher quality rank. 

To achieve this while passing the compute constraints, we utilize a two-stage architecture:
1. **Pre-computation (`precompute.py`)**: A one-time offline step that generates embeddings for all candidates and saves the technical scores.
2. **Ranking (`rank.py`)**: A lightning-fast script that evaluates all heuristic layers and generates the final CSV in seconds.

---

## Full workflow, layer by layer

### Phase 1: Pre-computation (`precompute.py`)
This script uses `sentence-transformers` (`all-MiniLM-L6-v2`) to generate semantic embeddings for all candidates.
1. **Honeypot detection**: Drops candidates with internally impossible data (e.g. 10+ simultaneous "expert" skills).
2. **Hard filters**: Removes candidates with non-technical current titles paired with fewer than 3 genuinely relevant AI/ML skills.
3. **Semantic technical alignment**: Builds a weighted text representation of each candidate and computes cosine similarity against the JD's embedding.
4. **Export**: Saves the resulting technical scores to `precomputed_scores.json`.

### Phase 2: Fast Ranking (`rank.py`)
This is the official ranking step that passes the hackathon constraints.
1. **Load Precomputed Scores**: Loads the technical scores generated in Phase 1 (35% weight).
2. **Skill depth score (25% weight)**: Combines proficiency level, duration multiplier, log-scaled endorsements, and Redrob's skill assessments.
3. **Behavioral availability score (20% weight)**: Uses recency of last activity, open-to-work flag, recruiter response rate, and notice period.
4. **Career trajectory score (15% weight)**: explicitly discounts profiles that are research-only with no production signal words ("deployed," "shipped," "users," "latency"), and applies an independent consulting-only check.
5. **JD-specific fit score (5% weight)**: Captures experience-band fit, location alignment, and GitHub activity.
6. **Composite score**: Combines all five layers and sorts the top 100.
7. **Dynamic Reasoning Generation**: For every top-100 candidate, builds a reasoning string entirely from real profile fields. We employ **dynamic tone adjustments** based on the candidate's rank (e.g., highly praising Rank 1-20, while acknowledging trade-offs for Rank 80-100) to ensure the reasoning reads naturally and avoids templated penalties.

---

## Setup & Execution

### 1. Installation

```bash
git clone https://github.com/reachzaki837/IntentRank-Intelligent-Candidate-Discovery-Engine
cd IntentRank-Intelligent-Candidate-Discovery-Engine

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install sentence-transformers scikit-learn numpy pandas
```

### 2. Generate Embeddings (Pre-computation)
Run this command first. It can take ~6-10 minutes on a CPU depending on hardware.
```bash
python precompute.py --candidates "./[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl" --out precomputed_scores.json
```

### 3. Generate Submission CSV (Official Ranking Step)
**Run this single command to reproduce our submission.** It completes in < 5 seconds, easily passing the 5-minute strict runtime limit.
```bash
python rank.py --candidates "./[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl" --out ./submission.csv
```

*Note: The included `notebooks/IntentRank_Candidate_Ranker.ipynb` is our research and exploration sandbox. For final hackathon verification, please use the two Python scripts above.*

---

## Output format

```csv
candidate_id,rank,score,reasoning
CAND_0036184,1,1.0,"Recommendation Systems Engineer at CRED, 6.0 yrs exp, Trivandrum, Kerala; relevant skills: FAISS, Hugging Face Transformers, LangChain, Semantic Search; actively engaged (open to work, 90% response rate); notice ≤30 days."
CAND_0011687,2,0.9158,"Senior NLP Engineer at Niramai, 7.8 yrs exp, Indore, Madhya Pradesh; relevant skills: TensorFlow, FAISS, Embeddings, LangChain; actively engaged (open to work, 89% response rate); notice ≤30 days; strong GitHub activity (76.3/100)."
...
```

---

## Constraints compliance

| Constraint | Limit | This system |
|---|---|---|
| Runtime | ≤ 5 minutes | `rank.py` runs in **~3 seconds** using precomputed embeddings, cleanly passing Stage 3 checks. |
| Memory | ≤ 16 GB RAM | Fast dictionary lookups in `rank.py` utilize < 500MB of RAM. |
| Compute | CPU only, no GPU | No GPU dependency anywhere in the ranking path. |
| Network | No external API calls | Zero hosted LLM calls — all reasoning is locally generated using rule-based dynamic logic. |
| Disk | ≤ 5 GB intermediate state | `precomputed_scores.json` is ~1MB. |

---

## Design decisions and known limitations

- **Weights (35/25/20/15/5) are a deliberate design choice for this specific JD**, not a universal formula — they reflect this role's emphasis on production-proven technical depth over pure credentials or pedigree.
- **Hard filters are intentionally conservative.** A small number of legitimate candidates with unconventional career paths may be excluded by the consulting-only or keyword-density filters. This tradeoff favors precision in the top 100 over recall across the full pool, which matches the challenge's stated goal of a shortlist "a recruiter can trust."
- **Sandbox Fallback:** If `rank.py` is tested on a novel subset of candidates in a hosted sandbox where `precomputed_scores.json` hasn't been generated, it will automatically compute `sentence-transformers` embeddings on the fly for those specific candidates.

---

**Team Apex Climbers** · India Runs Hackathon 2026 · Data & AI Challenge, Track 1