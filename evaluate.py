"""
evaluate.py
-----------
Offline Retrieval Evaluation Harness for hArI with Stratified Difficulty.
Benchmarks Supabase pgvector + tsvector RRF hybrid search against a 20-question Golden Dataset.

Computes:
    - Overall Hit Rate @ 3 & @ 5 (Recall)
    - Mean Reciprocal Rank (MRR)
    - Stratified Metrics by Difficulty (Easy / Medium / Hard)
    - Latency (milliseconds)
"""

import os
import time
from dotenv import load_dotenv
from supabase import create_client, ClientOptions
from sentence_transformers import SentenceTransformer

load_dotenv()

# ── 1. Configuration ──────────────────────────────────────────────────────────
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
TEST_EMAIL   = os.environ.get("TEST_USER_EMAIL")
TEST_PASS    = os.environ.get("TEST_USER_PASSWORD")

# ── 2. Normalized 20-Question Golden Dataset ─────────────────────────────────
GOLDEN_DATASET = [
    # ── 5 EASY (Direct Keywords & Acronyms) ──
    {
        "difficulty": "Easy",
        "query": "What is the token vocabulary size for the byte pair encoding in translation?",
        "expected_file": "Attention Is All You Need.pdf",
        "keywords": ["byte-pair", "vocabulary", "37000"]
    },
    {
        "difficulty": "Easy",
        "query": "What dataset was used for training English to German translation in transformers?",
        "expected_file": "Attention Is All You Need.pdf",
        "keywords": ["WMT 2014", "English-German", "dataset"]
    },
    {
        "difficulty": "Easy",
        "query": "What is StateGraph in LangGraph and what does it hold?",
        "expected_file": "Interview_Revision_Sheet_AI_Engineering.pdf",
        "keywords": ["StateGraph", "TypedDict", "single source of truth"]
    },
    {
        "difficulty": "Easy",
        "query": "What optimizer was used in the Transformer paper?",
        "expected_file": "Attention Is All You Need.pdf",
        "keywords": ["Adam", "beta1", "beta2"]
    },
    {
        "difficulty": "Easy",
        "query": "What are conditional edges in a state graph?",
        "expected_file": "Interview_Revision_Sheet_AI_Engineering.pdf",
        "keywords": ["conditional", "edges", "branch"]
    },

    # ── 8 MEDIUM (Paraphrased & Conceptual Mechanics) ──
    {
        "difficulty": "Medium",
        "query": "How does LangGraph manage stateful agent workflows using graph structures?",
        "expected_file": "Interview_Revision_Sheet_AI_Engineering.pdf",
        "keywords": ["LangGraph", "StateGraph", "stateful"]
    },
    {
        "difficulty": "Medium",
        "query": "What are reducers and how does node output merge into global state?",
        "expected_file": "Interview_Revision_Sheet_AI_Engineering.pdf",
        "keywords": ["Reducers", "state", "node"]
    },
    {
        "difficulty": "Medium",
        "query": "Why use graph orchestration over a plain chain or while loop for agents?",
        "expected_file": "Interview_Revision_Sheet_AI_Engineering.pdf",
        "keywords": ["deterministic", "routing", "persistence"]
    },
    {
        "difficulty": "Medium",
        "query": "Why does multi-head attention perform better than a single attention function?",
        "expected_file": "Attention Is All You Need.pdf",
        "keywords": ["multi-head", "subspaces", "representation"]
    },
    {
        "difficulty": "Medium",
        "query": "How does human in the loop interrupts work in agent orchestration?",
        "expected_file": "Interview_Revision_Sheet_AI_Engineering.pdf",
        "keywords": ["human-in-the-loop", "interrupts", "checkpointing"]
    },
    {
        "difficulty": "Medium",
        "query": "How do self-attention layers compare to recurrent layers in computational complexity?",
        "expected_file": "Attention Is All You Need.pdf",
        "keywords": ["complexity", "sequential", "operations", "O(1)"]
    },
    {
        "difficulty": "Medium",
        "query": "What handles agent replanning and retries in graph architectures?",
        "expected_file": "Interview_Revision_Sheet_AI_Engineering.pdf",
        "keywords": ["cycles", "loops", "retry"]
    },
    {
        "difficulty": "Medium",
        "query": "What is the role of residual connections and layer normalization in transformer blocks?",
        "expected_file": "Attention Is All You Need.pdf",
        "keywords": ["LayerNorm", "residual", "sub-layer"]
    },

    # ── 7 HARD (Abstract, Multi-Term, & Edge Architectural Queries) ──
    {
        "difficulty": "Hard",
        "query": "Why is the dot product scaled by the square root of key dimension in attention?",
        "expected_file": "Attention Is All You Need.pdf",
        "keywords": ["suspect", "gradients", "vanishing", "softmax"]
    },
    {
        "difficulty": "Hard",
        "query": "How does the decoder prevent positions from attending to subsequent future tokens?",
        "expected_file": "Attention Is All You Need.pdf",
        "keywords": ["masking", "subsequent", "prevent", "leftward"]
    },
    {
        "difficulty": "Hard",
        "query": "What is the mathematical formulation of positional encodings using sinusoidal functions?",
        "expected_file": "Attention Is All You Need.pdf",
        "keywords": ["sin", "cos", "frequencies", "10000"]
    },
    {
        "difficulty": "Hard",
        "query": "How do checkpointing and memory persistence ensure fault tolerance during long-running tasks?",
        "expected_file": "Interview_Revision_Sheet_AI_Engineering.pdf",
        "keywords": ["checkpointing", "persistence", "state"]
    },
    {
        "difficulty": "Hard",
        "query": "What dropout rate and label smoothing values were applied during model regularization?",
        "expected_file": "Attention Is All You Need.pdf",
        "keywords": ["dropout", "0.1", "label", "smoothing"]
    },
    {
        "difficulty": "Hard",
        "query": "How can an agent workflow handle an answer that spans across multiple chunks?",
        "expected_file": "Interview_Revision_Sheet_AI_Engineering.pdf",
        "keywords": ["overlap", "metadata", "reranking"]
    },
    {
        "difficulty": "Hard",
        "query": "How does scaled dot product attention handle very large values of dimension d_k?",
        "expected_file": "Attention Is All You Need.pdf",
        "keywords": ["large", "magnitude", "extremely", "small"]
    }
]


# ── 3. Evaluation Runner ──────────────────────────────────────────────────────

def run_evaluation():
    print("=" * 72)
    print("🚀 hArI Hybrid Search (RRF) Benchmark: 20-Query Stratified Test")
    print("=" * 72)

    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY in .env file.")

    # 1. Authenticate with Supabase
    email = TEST_EMAIL or input("Enter your hArI login email: ").strip()
    password = TEST_PASS or input("Enter your hArI login password: ").strip()

    auth_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    auth_res = auth_client.auth.sign_in_with_password({"email": email, "password": password})
    jwt_token = auth_res.session.access_token

    supabase = create_client(
        SUPABASE_URL, 
        SUPABASE_KEY, 
        options=ClientOptions(headers={"Authorization": f"Bearer {jwt_token}"})
    )

    print("\n⏳ Loading Embedding Model (all-MiniLM-L6-v2)...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    difficulty_stats = {
        "Easy": {"hits_3": 0, "hits_5": 0, "rr": [], "total": 0},
        "Medium": {"hits_3": 0, "hits_5": 0, "rr": [], "total": 0},
        "Hard": {"hits_3": 0, "hits_5": 0, "rr": [], "total": 0}
    }

    latencies = []

    print(f"\nRunning benchmark across {len(GOLDEN_DATASET)} queries...\n")
    print(f"{'#':<3} | {'Diff':<6} | {'Query':<42} | {'Rank':<6} | {'Latency':<8}")
    print("-" * 72)

    for i, test in enumerate(GOLDEN_DATASET, start=1):
        diff = test["difficulty"]
        query = test["query"]
        expected_file = test["expected_file"]
        keywords = test["keywords"]

        difficulty_stats[diff]["total"] += 1

        # Measure latency
        start_time = time.perf_counter()
        
        query_embedding = embedder.encode(query, convert_to_numpy=True).tolist()

        rpc_res = supabase.rpc("hybrid_search", {
            "query_text": query,
            "query_embedding": query_embedding,
            "match_count": 5
        }).execute()
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        latencies.append(elapsed_ms)

        # Check rank position
        rank_found = 0
        for rank, row in enumerate(rpc_res.data, start=1):
            content = row.get("content", "")
            meta = row.get("metadata") or {}
            source = meta.get("filename") or meta.get("source") or row.get("filename")

            file_match = (source == expected_file)
            keyword_match = any(kw.lower() in content.lower() for kw in keywords)

            if file_match and keyword_match:
                rank_found = rank
                break

        # Record metrics
        if rank_found > 0:
            if rank_found <= 3:
                difficulty_stats[diff]["hits_3"] += 1
            if rank_found <= 5:
                difficulty_stats[diff]["hits_5"] += 1
            difficulty_stats[diff]["rr"].append(1.0 / rank_found)
            rank_str = f"#{rank_found}"
        else:
            difficulty_stats[diff]["rr"].append(0.0)
            rank_str = "MISS"

        short_q = query[:39] + "..." if len(query) > 39 else query
        print(f"{i:<3} | {diff:<6} | {short_q:<42} | {rank_str:<6} | {elapsed_ms:>6.1f}ms")

    # ── 4. Summary Scorecard ──────────────────────────────────────────────────
    n = len(GOLDEN_DATASET)
    total_hits_3 = sum(d["hits_3"] for d in difficulty_stats.values())
    total_hits_5 = sum(d["hits_5"] for d in difficulty_stats.values())
    all_rr = [rr for d in difficulty_stats.values() for rr in d["rr"]]

    overall_hit_3 = (total_hits_3 / n) * 100
    overall_hit_5 = (total_hits_5 / n) * 100
    overall_mrr   = sum(all_rr) / n
    avg_latency   = sum(latencies) / n

    print("\n" + "=" * 72)
    print("📊 BENCHMARK SUMMARY SCORECARD")
    print("=" * 72)
    print(f"Overall Hit Rate @ 3 (Recall @ 3) : {overall_hit_3:>6.1f}%")
    print(f"Overall Hit Rate @ 5 (Recall @ 5) : {overall_hit_5:>6.1f}%")
    print(f"Overall Mean Reciprocal Rank (MRR): {overall_mrr:>6.3f} (Max: 1.000)")
    print(f"Average Retrieval Latency         : {avg_latency:>6.1f} ms")
    print("-" * 72)
    print("DIFFICULTY BREAKDOWN:")
    for diff, data in difficulty_stats.items():
        diff_count = data["total"]
        diff_hit_5 = (data["hits_5"] / diff_count) * 100 if diff_count else 0
        diff_mrr = sum(data["rr"]) / diff_count if diff_count else 0
        print(f"  • {diff:<6} ({diff_count} queries) -> Hit Rate @ 5: {diff_hit_5:>5.1f}% | MRR: {diff_mrr:.3f}")
    print("=" * 72)


if __name__ == "__main__":
    run_evaluation()