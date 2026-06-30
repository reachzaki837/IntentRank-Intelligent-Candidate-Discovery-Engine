import json
import math
import csv
import re
import argparse
import hashlib
from datetime import datetime, date
from pathlib import Path
import pandas as pd

# --- Constants & Setup ---
REFERENCE_DATE = date(2026, 6, 29)

def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default

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
PREFERRED_LOCATIONS = {"pune", "noida", "hyderabad", "mumbai", "delhi", "bangalore", "bengaluru", "india"}
GOOD_TITLE_SIGNALS = {
    "machine learning", "ml engineer", "ai engineer", "data scientist",
    "nlp engineer", "research engineer", "applied scientist", "search engineer",
    "ranking engineer", "recommendation", "backend engineer", "software engineer",
    "platform engineer", "data engineer", "mlops", "senior engineer",
    "staff engineer", "founding engineer", "tech lead", "principal engineer",
}

def _is_consulting_company(company_name: str) -> bool:
    return bool(_CF_PATTERN.search(company_name))

def is_honeypot(candidate: dict) -> bool:
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

def score_skill_depth(candidate: dict) -> float:
    skills = candidate.get("skills", [])
    sig = candidate.get("redrob_signals", {})
    raw_assessment = sig.get("skill_assessment_scores", {}) or {}
    assessment_scores = {k.lower(): v for k, v in raw_assessment.items()}
    depth_total = 0.0
    relevant_count = 0
    for sk in skills:
        name_lower = sk.get("name", "").lower()
        if not any(ds in name_lower for ds in AI_DEPTH_SKILLS): continue
        relevant_count += 1
        proficiency = sk.get("proficiency", "")
        dur = sk.get("duration_months", 0) or 0
        endorsements = sk.get("endorsements", 0) or 0
        base = {"expert": 1.0, "advanced": 0.75, "intermediate": 0.4}.get(proficiency, 0.15)
        dur_mult = min(1.0, dur / 24.0) if dur > 0 else 0.1
        endorsement_bonus = min(0.3, math.log1p(endorsements) / 15.0)
        assess_bonus = 0.0
        if name_lower in assessment_scores:
            assess_bonus = (_safe_float(assessment_scores[name_lower]) / 100.0) * 0.2
        depth_total += base * dur_mult + endorsement_bonus + assess_bonus
    if relevant_count == 0: return 0.0
    return min(1.0, depth_total / 10.0)

def score_behavioral_availability(candidate: dict) -> float:
    sig = candidate.get("redrob_signals", {})
    score = 0.0
    last_active_s = sig.get("last_active_date", "")
    if last_active_s:
        try:
            last_active = datetime.strptime(last_active_s, "%Y-%m-%d").date()
            days_ago = (REFERENCE_DATE - last_active).days
            if days_ago <= 30: score += 0.30
            elif days_ago <= 90: score += 0.22
            elif days_ago <= 180: score += 0.12
            else: score += 0.02
        except ValueError:
            score += 0.05
    if sig.get("open_to_work_flag"): score += 0.20
    score += _safe_float(sig.get("recruiter_response_rate", 0.0)) * 0.25
    score += _safe_float(sig.get("interview_completion_rate", 0.0)) * 0.15
    notice = sig.get("notice_period_days", 90)
    if notice <= 30: score += 0.10
    elif notice <= 60: score += 0.05
    else: score += 0.01
    return min(1.0, score)

def score_career_trajectory(candidate: dict) -> float:
    career = candidate.get("career_history", [])
    if not career: return 0.3
    def sort_key(h): return "9999-99-99" if h.get("is_current") else h.get("start_date", "0000-00-00")
    sorted_career = sorted(career, key=sort_key, reverse=True)
    def role_relevance(hist):
        text = (hist.get("title", "") + " " + hist.get("description", "")).lower()
        ml_hits = sum(1 for ds in AI_DEPTH_SKILLS if ds in text)
        title_hits = sum(1 for gt in GOOD_TITLE_SIGNALS if gt in text)
        return min(1.0, ml_hits * 0.1 + title_hits * 0.2)
    role_scores = [role_relevance(h) for h in sorted_career]
    weights = [1.0 / (1.0 + 0.5 * i) for i in range(len(role_scores))]
    trajectory = sum(s * w for s, w in zip(role_scores, weights)) / sum(weights)
    all_text = " ".join(h.get("description", "") for h in career).lower()
    production_hits = sum(1 for p in ["production", "deployed", "shipped", "users", "traffic", "scale", "latency", "throughput", "api", "service", "system", "platform"] if re.search(r'\b' + re.escape(p) + r'\b', all_text))
    research_hits = sum(1 for r in ["lab", "research", "paper", "academic", "thesis", "phd", "university", "professor", "publish"] if re.search(r'\b' + re.escape(r) + r'\b', all_text))
    if research_hits > production_hits and production_hits < 3:
        trajectory *= 0.6
    if len(career) > 1:
        avg_duration = sum(h.get("duration_months", 12) for h in career) / len(career)
        if avg_duration < 12: trajectory *= 0.85
    if all(_is_consulting_company(h.get("company", "")) for h in career):
        trajectory *= 0.5
    return min(1.0, trajectory)

def score_jd_specific(candidate: dict) -> float:
    profile = candidate.get("profile", {})
    sig = candidate.get("redrob_signals", {})
    score = 0.0
    yoe = profile.get("years_of_experience", 0)
    if 5 <= yoe <= 9: score += 0.40
    elif 4 <= yoe < 5 or 9 < yoe <= 12: score += 0.25
    elif 3 <= yoe < 4 or 12 < yoe <= 15: score += 0.10
    else: score += 0.02
    loc = profile.get("location", "").lower()
    country = profile.get("country", "").lower()
    if any(city in loc for city in PREFERRED_LOCATIONS) or country == "india": score += 0.30
    elif sig.get("willing_to_relocate"): score += 0.20
    else: score += 0.05
    gh = sig.get("github_activity_score", -1)
    if gh is not None and gh >= 0: score += min(0.30, gh / 100.0 * 0.30)
    return min(1.0, score)

WEIGHTS = {
    "technical": 0.35,
    "skill_depth": 0.25,
    "behavioral": 0.20,
    "trajectory": 0.15,
    "jd_specific": 0.05,
}

def composite_score(scores: dict) -> float:
    return sum(WEIGHTS[k] * scores[k] for k in WEIGHTS)

def generate_reasoning_dynamic(candidate: dict, scores: dict, rank: int, raw_score: float) -> str:
    profile = candidate.get("profile", {})
    skills = candidate.get("skills", [])
    sig = candidate.get("redrob_signals", {})

    title = profile.get("current_title", "Unknown")
    company = profile.get("current_company", "")
    yoe = profile.get("years_of_experience", 0)
    
    top_skills = [
        s["name"] for s in skills
        if any(ds in s.get("name", "").lower() for ds in AI_DEPTH_SKILLS)
        and s.get("proficiency") in ("advanced", "expert")
    ][:4]
    if len(top_skills) < 4:
        intermediate_skills = [
            s["name"] for s in skills
            if any(ds in s.get("name", "").lower() for ds in AI_DEPTH_SKILLS)
            and s.get("proficiency") == "intermediate"
            and s["name"] not in top_skills
        ][:2]
    else:
        intermediate_skills = []

    rr = _safe_float(sig.get("recruiter_response_rate", 0))
    open_work = sig.get("open_to_work_flag", False)
    notice = sig.get("notice_period_days", 90)

    # Use hash for deterministic variations
    hash_val = int(hashlib.md5(candidate["candidate_id"].encode()).hexdigest(), 16)
    variant = hash_val % 3
    
    parts = []
    
    # 1. Opening statement (changes based on rank)
    if rank <= 20:
        openings = [
            f"Exceptional fit for JD requirements: {yoe:.1f} years of experience.",
            f"Top-tier candidate currently working as {title}.",
            f"Highly relevant {title} with {yoe:.1f} YOE in production ML."
        ]
        parts.append(openings[variant])
    elif rank <= 60:
        openings = [
            f"Strong technical match currently at {company or 'a tech firm'}.",
            f"Solid {title} with {yoe:.1f} YOE.",
            f"Demonstrates clear JD alignment with {yoe:.1f} years experience."
        ]
        parts.append(openings[variant])
    else:
        openings = [
            f"Viable candidate with {yoe:.1f} YOE as {title}.",
            f"Good baseline technical fit currently at {company or 'industry'}.",
            f"Shows potential for the role with {yoe:.1f} YOE."
        ]
        parts.append(openings[variant])

    # 2. Skill statements connecting to JD
    if top_skills:
        skill_stmts = [
            f"Matches core JD needs with expertise in {', '.join(top_skills)}.",
            f"Brings strong hands-on skills in {', '.join(top_skills)}.",
            f"Technical depth aligns with JD (features {', '.join(top_skills)})."
        ]
        parts.append(skill_stmts[variant])
    elif intermediate_skills:
        parts.append(f"Developing proficiency in JD requirements like {', '.join(intermediate_skills)}.")

    # 3. Honest concerns or strengths
    if open_work and rr >= 0.5:
        parts.append("Positive engagement signals.")
    elif rr < 0.2:
        parts.append("Warning: Low response rate indicates availability risk.")
        
    if notice > 60:
        parts.append("Note: Notice period exceeds 60 days, risking project timelines.")
    elif notice <= 30:
        parts.append("Available to join quickly (≤30 days notice).")
        
    if rank > 50 and any(bt in (title or "").lower() for bt in BAD_PRIMARY_TITLES):
        parts.append(f"Title ({title}) is non-traditional for ML, requiring deeper vetting.")

    reasoning = " ".join(parts)
    return reasoning[:250] if len(reasoning) <= 250 else reasoning[:247] + "..."

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=str, required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", type=str, required=True, help="Path for output submission.csv")
    parser.add_argument("--precomputed", type=str, default="precomputed_scores.json", help="Path to precomputed embeddings scores")
    args = parser.parse_args()
    
    candidates_path = Path(args.candidates)
    precomp_path = Path(args.precomputed)
    
    if not precomp_path.exists():
        raise FileNotFoundError(f"Precomputed scores not found at {precomp_path}. Run precompute.py first.")
        
    with open(precomp_path, "r", encoding="utf-8") as f:
        precomputed_scores = json.load(f)

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

    all_results = []
    for cand in kept:
        cid = cand["candidate_id"]
        # If running a small sample sandbox test, the candidate might not be in the precomputed file!
        # Wait, the spec says the sandbox must accept a small sample and run end-to-end. 
        # If we just do lookups, a NEW sandbox sample will fail because it's not in the precomputed JSON.
        # So rank.py should fallback to live embeddings if the score isn't found!
        tech_score = precomputed_scores.get(cid)
        
        if tech_score is None:
            # Fallback to computing on the fly for sandbox test
            try:
                from sentence_transformers import SentenceTransformer
                embed_model = SentenceTransformer("all-MiniLM-L6-v2")
                jd_emb = embed_model.encode([JD_CORE_CONCEPT_TEXT], normalize_embeddings=True)
                c_text = build_candidate_text(cand)
                c_emb = embed_model.encode([c_text], normalize_embeddings=True)
                tech_score = float((c_emb @ jd_emb.T)[0][0])
            except ImportError:
                tech_score = 0.5 # Safe fallback if model missing in strict environment
            
        scores = {
            "technical":   float(tech_score),
            "skill_depth": score_skill_depth(cand),
            "behavioral":  score_behavioral_availability(cand),
            "trajectory":  score_career_trajectory(cand),
            "jd_specific": score_jd_specific(cand),
        }
        final = composite_score(scores)
        all_results.append((final, cand, scores))

    all_results.sort(key=lambda x: (-x[0], x[1]["candidate_id"])) # Tiebreaker by candidate_id
    top_100 = all_results[:100]

    if not top_100:
        print("No candidates found!")
        return

    max_score = top_100[0][0]
    min_score = top_100[-1][0]
    score_range = max_score - min_score if max_score > min_score else 1.0

    output_rows = []
    for i, (raw_score, cand, scores) in enumerate(top_100):
        rank = i + 1
        norm_score = 0.20 + 0.80 * (raw_score - min_score) / score_range
        norm_score = round(norm_score - i * 1e-8, 6)
        reasoning = generate_reasoning_dynamic(cand, scores, rank, raw_score)
        output_rows.append({
            "candidate_id": cand["candidate_id"],
            "rank": rank,
            "score": round(norm_score, 4),
            "reasoning": reasoning,
        })

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Successfully ranked top 100 candidates and wrote to {args.out}")

def build_candidate_text(candidate: dict) -> str:
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

if __name__ == "__main__":
    main()
