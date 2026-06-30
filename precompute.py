import json
import math
import re
import time
from datetime import datetime, date
from pathlib import Path
from collections import Counter
import numpy as np
import argparse

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Warning: sentence-transformers not found. Please pip install sentence-transformers")
    raise

# --- Constants & Setup ---
JD_CORE_CONCEPT_TEXT = (
    "We are hiring for a role centered on embeddings, retrieval, and ranking systems. "
    "The ideal candidate has built and deployed semantic search or recommendation systems "
    "to production, using vector databases such as Pinecone, Weaviate, Qdrant, FAISS, or "
    "OpenSearch, and understands hybrid search combining dense and sparse retrieval. "
    "Experience with evaluation frameworks (NDCG, MRR, MAP, offline and online A/B testing) "
    "is highly valued. Familiarity with fine-tuning language models, LoRA or QLoRA, and "
    "applied NLP in a production environment is a strong plus. We want someone who has "
    "shipped real systems serving real users, not only academic or research experiments."
)

BAD_PRIMARY_TITLES = {
    "marketing manager", "content writer", "graphic designer", "hr manager",
    "accountant", "sales executive", "customer support", "operations manager",
    "project manager", "mechanical engineer", "civil engineer", "business analyst",
}

CONSULTING_FIRMS = {
    "wipro", "infosys", "tcs", "accenture", "cognizant", "capgemini", "hcl",
    "tech mahindra", "mphasis", "hexaware", "ltimindtree", "mindtree",
    "l&t infotech",
}

_CF_PATTERN = re.compile(
    r'\b(?:' + '|'.join(re.escape(cf) for cf in CONSULTING_FIRMS) + r')\b',
    re.IGNORECASE
)

AI_DEPTH_SKILLS = {
    "embeddings", "vector search", "semantic search", "dense retrieval",
    "fine-tuning llms", "fine tuning llms", "llm fine-tuning", "rag",
    "retrieval augmented generation", "nlp", "transformers", "bert",
    "sentence transformers", "pytorch", "tensorflow", "scikit-learn", "sklearn",
    "faiss", "pinecone", "weaviate", "qdrant", "chromadb", "langchain",
    "llamaindex", "openai", "anthropic", "information retrieval", "ranking",
    "recommendation systems", "evaluation", "ndcg", "a/b testing", "mlops",
    "hugging face", "huggingface",
}

def _is_consulting_company(company_name: str) -> bool:
    """
    Checks if a company name contains a known consulting firm as a whole word.
    Used to discount candidates with purely consulting/staffing backgrounds.
    """
    return bool(_CF_PATTERN.search(company_name))

def is_honeypot(candidate: dict) -> bool:
    """
    Returns True if the candidate profile contains internally impossible data,
    such as claiming expert proficiency with 0 months of experience.
    """
    skills = candidate.get("skills", [])
    for sk in skills:
        if sk.get("proficiency") in ("advanced", "expert") and sk.get("duration_months", 1) == 0:
            return True
    expert_count = sum(1 for s in skills if s.get("proficiency") == "expert")
    if expert_count >= 10:
        return True
    for hist in candidate.get("career_history", []):
        start_s, end_s = hist.get("start_date"), hist.get("end_date")
        claimed = hist.get("duration_months", 0)
        if start_s and end_s and claimed > 0:
            try:
                start = datetime.strptime(start_s, "%Y-%m-%d").date()
                end = datetime.strptime(end_s, "%Y-%m-%d").date()
                actual_months = (end.year - start.year) * 12 + (end.month - start.month)
                if claimed > actual_months + 6:
                    return True
            except ValueError:
                pass
    return False

def hard_filter(candidate: dict) -> tuple[bool, str]:
    """
    Applies aggressive heuristic filters to drop low-quality candidates before
    expensive embedding computation. Returns (keep_boolean, exclusion_reason).
    """
    if is_honeypot(candidate):
        return False, "honeypot"
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    current_title = (profile.get("current_title") or "").lower()
    if career:
        all_consulting = all(_is_consulting_company(h.get("company", "")) for h in career)
        is_bad_title = any(bt in current_title for bt in BAD_PRIMARY_TITLES)
        if all_consulting and is_bad_title:
            return False, "consulting-only, non-technical career"
    is_bad_title = any(bt in current_title for bt in BAD_PRIMARY_TITLES)
    if is_bad_title:
        skills_text = " ".join(s.get("name", "").lower() for s in candidate.get("skills", []))
        ai_skill_count = sum(1 for ds in AI_DEPTH_SKILLS if ds in skills_text)
        if ai_skill_count < 3:
            return False, f"non-technical title ({current_title}) with shallow AI skill listing"
    return True, ""

def build_candidate_text(candidate: dict) -> str:
    """
    Constructs a weighted text representation of a candidate's profile for semantic matching.
    Current roles and high-confidence skills are given more textual weight.
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    parts = [profile.get("headline", ""), profile.get("summary", "")]
    for hist in career:
        weight = 3 if hist.get("is_current") else (2 if hist.get("duration_months", 0) > 12 else 1)
        desc = hist.get("description", "") + " " + hist.get("title", "")
        parts.extend([desc] * weight)
    for sk in skills:
        name = sk.get("name", "")
        if sk.get("proficiency") in ("advanced", "expert") and sk.get("duration_months", 0) > 6:
            parts.append(name + " " + name)
        else:
            parts.append(name)
    return " ".join(parts).lower().strip()

def main():
    """
    Main execution pipeline for Phase 1: Pre-computation.
    Loads candidates, applies hard filters, generates semantic embeddings,
    and exports the technical alignment scores to a JSON file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=str, default="./candidates.jsonl")
    parser.add_argument("--out", type=str, default="precomputed_scores.json")
    args = parser.parse_args()
    
    candidates_path = Path(args.candidates)
    if not candidates_path.exists():
        raise FileNotFoundError(f"Candidates file not found at {candidates_path}")

    print(f"Loading candidates from {candidates_path}...")
    candidates = []
    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    
    kept = []
    for c in candidates:
        keep, _ = hard_filter(c)
        if keep:
            kept.append(c)
    print(f"Kept {len(kept)} / {len(candidates)} candidates.")

    print("Loading SentenceTransformer model...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    jd_embedding = embed_model.encode([JD_CORE_CONCEPT_TEXT], normalize_embeddings=True)

    print("Encoding candidate texts in batches...")
    start_time = time.time()
    batch_size = 256
    results = {}

    for batch_start in range(0, len(kept), batch_size):
        batch = kept[batch_start: batch_start + batch_size]
        candidate_texts = [build_candidate_text(c) for c in batch]
        
        non_empty_idx = [i for i, t in enumerate(candidate_texts) if t.strip()]
        sims = np.zeros(len(candidate_texts))
        if non_empty_idx:
            texts_to_encode = [candidate_texts[i] for i in non_empty_idx]
            embeddings = embed_model.encode(texts_to_encode, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
            sim_scores = embeddings @ jd_embedding.T
            for idx, score in zip(non_empty_idx, sim_scores.flatten()):
                sims[idx] = float(score)
        
        for cand, sim in zip(batch, sims):
            results[cand["candidate_id"]] = sim
        
        if batch_start % 5000 == 0:
            elapsed = time.time() - start_time
            print(f"Scored {batch_start + len(batch)}/{len(kept)} | elapsed {elapsed:.1f}s")
            
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f)
    
    elapsed = time.time() - start_time
    print(f"Finished precomputing embeddings in {elapsed:.1f}s. Saved to {args.out}")

if __name__ == "__main__":
    main()
