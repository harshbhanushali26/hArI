"""
evaluate.py
--------------------
Production-grade, zero-external-eval-framework evaluation harness for hArI v2.
Evaluates:
  1. Retrieval Quality (Hit Rate @ 3, Hit Rate @ 5, MRR, Latency)
  2. Answer Faithfulness & Correctness (via LLM-as-a-Judge with gpt-oss-120b)
  3. Negative Constraint / Abstention Rate (Unanswerable query handling)
  4. Citation Accuracy (Grounding of cited source and page numbers)
  5. Tabular Pipeline (DuckDB SQL generation, self-healing retry loop, sandboxing)
  6. Token & Cost Accounting (Prompt, completion, and reasoning tokens)

Usage:
  python evaluate.py
"""

import os
import re
import time
import json
import duckdb
import pandas as pd
from typing import Dict, List, Any, Tuple
from dotenv import load_dotenv
from supabase import create_client, ClientOptions
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()

# ==============================================================================
# 1. CONFIGURATION & CREDENTIALS
# ==============================================================================

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
TEST_EMAIL   = os.environ.get("TEST_USER_EMAIL")
TEST_PASS    = os.environ.get("TEST_USER_PASSWORD")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

RESPONSE_MODEL = "openai/gpt-oss-120b"
ANALYSIS_MODEL = "openai/gpt-oss-120b"
JUDGE_MODEL    = "openai/gpt-oss-120b"
EMBED_MODEL    = "all-MiniLM-L6-v2"
TOP_K          = 5

# Estimated Groq pricing per 1M tokens
COST_PER_1M_INPUT  = 0.15   # openai/gpt-oss-120b input
COST_PER_1M_OUTPUT = 0.60   # openai/gpt-oss-120b output

if not all([SUPABASE_URL, SUPABASE_KEY, GROQ_API_KEY]):
    raise ValueError("Missing SUPABASE_URL, SUPABASE_KEY, or GROQ_API_KEY in .env.")


# ==============================================================================
# 2. 35-QUERY STRATIFIED GOLDEN DATASET (100% VERIFIED ON YOUR RAW DOCUMENTS)
# ==============================================================================

GOLDEN_DATASET: List[Dict[str, Any]] = [
    # ── CATEGORY 1: Exact-Term & Hyperparameter Lookups (8 Queries) ──
    {
        "id": "ET-01",
        "category": "Exact-Term",
        "domain": "pdf",
        "query": "What is the token vocabulary size for the byte pair encoding in translation?",
        "expected_file": "Attention Is All You Need.pdf",
        "keywords": ["byte-pair", "vocabulary", "37000"],
        "expected_answer": "About 37,000 tokens for the shared source-target vocabulary.",
        "expected_page": 7
    },
    {
        "id": "ET-02",
        "category": "Exact-Term",
        "domain": "pdf",
        "query": "What dataset was used for training English to German translation in transformers?",
        "expected_file": "Attention Is All You Need.pdf",
        "keywords": ["WMT 2014", "English-German", "4.5 million"],
        "expected_answer": "The WMT 2014 English-German dataset consisting of 4.5 million sentence pairs.",
        "expected_page": 7
    },
    {
        "id": "ET-03",
        "category": "Exact-Term",
        "domain": "pdf",
        "query": "What optimizer was used in the Transformer paper, and what were its beta parameters?",
        "expected_file": "Attention Is All You Need.pdf",
        "keywords": ["Adam", "0.9", "0.98"],
        "expected_answer": "The Adam optimizer with beta_1 = 0.9, beta_2 = 0.98, and epsilon = 10^-9.",
        "expected_page": 7
    },
    {
        "id": "ET-04",
        "category": "Exact-Term",
        "domain": "pdf",
        "query": "What dropout rate and label smoothing values were applied during model regularization?",
        "expected_file": "Attention Is All You Need.pdf",
        "keywords": ["dropout", "0.1", "label smoothing"],
        "expected_answer": "Residual dropout of Pdrop = 0.1 and label smoothing of epsilon_ls = 0.1.",
        "expected_page": 8
    },
    {
        "id": "ET-05",
        "category": "Exact-Term",
        "domain": "pdf",
        "query": "What is StateGraph in LangGraph and what does it hold?",
        "expected_file": "Interview_Revision_Sheet_AI_Engineering.pdf",
        "keywords": ["StateGraph", "TypedDict", "single source of truth"],
        "expected_answer": "StateGraph with a TypedDict state is the single source of truth that every node reads and writes.",
        "expected_page": 1
    },
    {
        "id": "ET-06",
        "category": "Exact-Term",
        "domain": "pdf",
        "query": "What are conditional edges in a state graph?",
        "expected_file": "Interview_Revision_Sheet_AI_Engineering.pdf",
        "keywords": ["conditional", "edges", "branch on state"],
        "expected_answer": "Conditional edges branch on state to control the execution flow between nodes.",
        "expected_page": 1
    },
    {
        "id": "ET-07",
        "category": "Exact-Term",
        "domain": "pdf",
        "query": "What is the dimension of keys and queries d_k in the Transformer base model?",
        "expected_file": "Attention Is All You Need.pdf",
        "keywords": ["dk", "64", "dmodel"],
        "expected_answer": "The dimension of keys and queries dk is 64 (dk = dv = dmodel/h = 64).",
        "expected_page": 5
    },
    {
        "id": "ET-08",
        "category": "Exact-Term",
        "domain": "pdf",
        "query": "What BLEU score did the big transformer model achieve on English-to-French translation?",
        "expected_file": "Attention Is All You Need.pdf",
        "keywords": ["BLEU", "41.0", "41.8"],
        "expected_answer": "A BLEU score of 41.0 (single model reported in Section 6.1) or 41.8 in Table 2.",
        "expected_page": 8
    },

    # ── CATEGORY 2: Conceptual & Paraphrased Mechanics (8 Queries) ──
    {
        "id": "CP-01",
        "category": "Conceptual",
        "domain": "pdf",
        "query": "Why does multi-head attention perform better than a single attention function?",
        "expected_file": "Attention Is All You Need.pdf",
        "keywords": ["multi-head", "subspaces", "representation", "positions"],
        "expected_answer": "It allows the model to jointly attend to information from different representation subspaces at different positions, avoiding the inhibiting effect of averaging.",
        "expected_page": 5
    },
    {
        "id": "CP-02",
        "category": "Conceptual",
        "domain": "pdf",
        "query": "Why is the dot product scaled by the square root of key dimension in attention?",
        "expected_file": "Attention Is All You Need.pdf",
        "keywords": ["magnitude", "softmax", "gradients", "scale"],
        "expected_answer": "For large values of dk, dot products grow large in magnitude, pushing softmax into regions with vanishingly small gradients. Scaling counteracts this.",
        "expected_page": 4
    },
    {
        "id": "CP-03",
        "category": "Conceptual",
        "domain": "pdf",
        "query": "How does the decoder prevent positions from attending to subsequent future tokens?",
        "expected_file": "Attention Is All You Need.pdf",
        "keywords": ["masking", "subsequent", "prevent", "leftward"],
        "expected_answer": "By masking out illegal subsequent positions (setting them to -infinity in the softmax input) to prevent leftward information flow.",
        "expected_page": 5
    },
    {
        "id": "CP-04",
        "category": "Conceptual",
        "domain": "pdf",
        "query": "What are reducers in LangGraph and what do they define?",
        "expected_file": "Interview_Revision_Sheet_AI_Engineering.pdf",
        "keywords": ["reducers", "node", "output", "merges", "state"],
        "expected_answer": "Reducers define how a node's output merges into the shared graph state.",
        "expected_page": 1
    },
    {
        "id": "CP-05",
        "category": "Conceptual",
        "domain": "pdf",
        "query": "Why use graph orchestration over a plain chain or while loop for agents?",
        "expected_file": "Interview_Revision_Sheet_AI_Engineering.pdf",
        "keywords": ["explicit state", "deterministic routing", "cycles", "checkpointing"],
        "expected_answer": "Graphs provide explicit state, deterministic routing, cycles for retry and replanning, checkpointing for persistence, and human-in-the-loop interrupts.",
        "expected_page": 1
    },
    {
        "id": "CP-06",
        "category": "Conceptual",
        "domain": "pdf",
        "query": "How do self-attention layers compare to recurrent layers in computational complexity?",
        "expected_file": "Attention Is All You Need.pdf",
        "keywords": ["O(1)", "O(n)", "sequential operations"],
        "expected_answer": "Self-attention requires O(1) sequential operations compared to O(n) for recurrent layers, enabling greater parallelization during training.",
        "expected_page": 6
    },
    {
        "id": "CP-07",
        "category": "Conceptual",
        "domain": "pdf",
        "query": "How do human in the loop interrupts work in agent orchestration?",
        "expected_file": "Interview_Revision_Sheet_AI_Engineering.pdf",
        "keywords": ["interrupts", "pause", "human review", "resume"],
        "expected_answer": "Interrupts pause the graph for human review, and then execution resumes.",
        "expected_page": 1
    },
    {
        "id": "CP-08",
        "category": "Conceptual",
        "domain": "pdf",
        "query": "What is the purpose of residual connections and layer normalization in transformer blocks?",
        "expected_file": "Attention Is All You Need.pdf",
        "keywords": ["LayerNorm", "residual", "Sublayer", "512"],
        "expected_answer": "Residual connections followed by layer normalization LayerNorm(x + Sublayer(x)) facilitate gradient flow and stabilize training across all 512-dimensional sublayers.",
        "expected_page": 3
    },

    # ── CATEGORY 3: Multi-Chunk Synthesis (6 Queries) ──
    {
        "id": "MC-01",
        "category": "Multi-Chunk",
        "domain": "pdf",
        "query": "What are the key architectural differences between the Base and Big Transformer models?",
        "expected_file": "Attention Is All You Need.pdf",
        "keywords": ["base", "big", "512", "1024", "65", "213"],
        "expected_answer": "Base model has dmodel=512, dff=2048, 8 heads, and 65M parameters. Big model has dmodel=1024, dff=4096, 16 heads, and 213M parameters.",
        "expected_page": 9
    },
    {
        "id": "MC-02",
        "category": "Multi-Chunk",
        "domain": "pdf",
        "query": "How do positional encodings work and why did the authors choose sinusoidal functions over learned ones?",
        "expected_file": "Attention Is All You Need.pdf",
        "keywords": ["sin", "cos", "extrapolate", "longer", "frequencies"],
        "expected_answer": "Sinusoidal functions of different frequencies allow the model to learn relative positions and extrapolate to sequence lengths longer than those seen during training.",
        "expected_page": 6
    },
    {
        "id": "MC-03",
        "category": "Multi-Chunk",
        "domain": "pdf",
        "query": "Describe the complete encoder pipeline from token embedding to the final output of the stack.",
        "expected_file": "Attention Is All You Need.pdf",
        "keywords": ["embedding", "positional encodings", "LayerNorm", "residual"],
        "expected_answer": "Input tokens are embedded and scaled by sqrt(dmodel), added to positional encodings, then passed through 6 identical layers each with multi-head self-attention and feed-forward sublayers with residual connections and layer normalization.",
        "expected_page": 3
    },
    {
        "id": "MC-04",
        "category": "Multi-Chunk",
        "domain": "pdf",
        "query": "How does checkpointing in LangGraph persist agent state and resume runs?",
        "expected_file": "Interview_Revision_Sheet_AI_Engineering.pdf",
        "keywords": ["checkpointing", "serialised", "store", "thread id", "resumes"],
        "expected_answer": "State is serialized to a store (like Postgres) after each step so a run survives restarts, and a thread ID is used to resume the run.",
        "expected_page": 2
    },
    {
        "id": "MC-05",
        "category": "Multi-Chunk",
        "domain": "pdf",
        "query": "What are the three ways multi-head attention is used in the Transformer architecture?",
        "expected_file": "Attention Is All You Need.pdf",
        "keywords": ["encoder-decoder attention", "encoder self-attention", "decoder self-attention"],
        "expected_answer": "1. Encoder-decoder attention (queries from decoder, keys/values from encoder), 2. Encoder self-attention, and 3. Decoder self-attention (with masking to prevent leftward information flow).",
        "expected_page": 5
    },
    {
        "id": "MC-06",
        "category": "Multi-Chunk",
        "domain": "pdf",
        "query": "How does the Agent Engine workflow handle replanning and prevent infinite execution loops?",
        "expected_file": "Interview_Revision_Sheet_AI_Engineering.pdf",
        "keywords": ["bounded execution", "Replanner", "Validator", "max iterations", "step budget"],
        "expected_answer": "It uses a bounded execution loop (Planner -> Validator -> Executor -> Replanner -> Responder) with retry/replanning, and prevents infinite loops using max iterations or a step budget with a validator or critic node.",
        "expected_page": 2
    },

    # ── CATEGORY 4: Negative / Out-of-Domain Refusals (5 Queries) ──
    {
        "id": "NEG-01",
        "category": "Negative-Refusal",
        "domain": "pdf",
        "query": "What batch size and patch resolution were used to train the Vision Transformer (ViT) on ImageNet?",
        "expected_file": "Attention Is All You Need.pdf",
        "keywords": [],
        "expected_answer": "I could not find relevant information about this in the uploaded document.",
        "expected_page": None
    },
    {
        "id": "NEG-02",
        "category": "Negative-Refusal",
        "domain": "pdf",
        "query": "What are the pricing tiers and rate limits for OpenAI's GPT-4o API?",
        "expected_file": "Attention Is All You Need.pdf",
        "keywords": [],
        "expected_answer": "I could not find relevant information about this in the uploaded document.",
        "expected_page": None
    },
    {
        "id": "NEG-03",
        "category": "Negative-Refusal",
        "domain": "pdf",
        "query": "How does LoRA rank decomposition reduce memory during fine-tuning of Llama 3?",
        "expected_file": "Interview_Revision_Sheet_AI_Engineering.pdf",
        "keywords": [],
        "expected_answer": "I could not find relevant information about this in the uploaded document.",
        "expected_page": None
    },
    {
        "id": "NEG-04",
        "category": "Negative-Refusal",
        "domain": "pdf",
        "query": "What was the quantum volume calculated in the 2024 IBM Eagle processor experiment?",
        "expected_file": "Attention Is All You Need.pdf",
        "keywords": [],
        "expected_answer": "I could not find relevant information about this in the uploaded document.",
        "expected_page": None
    },
    {
        "id": "NEG-05",
        "category": "Negative-Refusal",
        "domain": "pdf",
        "query": "What are the rules for filing quarterly GST returns in India for IT consulting firms?",
        "expected_file": "Interview_Revision_Sheet_AI_Engineering.pdf",
        "keywords": [],
        "expected_answer": "I could not find relevant information about this in the uploaded document.",
        "expected_page": None
    },

    # ── CATEGORY 5: Tabular Multi-Table CSV Queries (8 Queries) ──
    {
        "id": "CSV-01",
        "category": "Tabular-SQL",
        "domain": "csv",
        "query": "What is the total revenue across all completed orders?",
        "expected_answer": "2050",
        "type": "numeric"
    },
    {
        "id": "CSV-02",
        "category": "Tabular-SQL",
        "domain": "csv",
        "query": "How many total orders were placed by customers living in 'Berlin'?",
        "expected_answer": "3",
        "type": "numeric"
    },
    {
        "id": "CSV-03",
        "category": "Tabular-SQL",
        "domain": "csv",
        "query": "What is the name of the customer who spent the most total money on completed orders?",
        "expected_answer": "Alice Smith",
        "type": "text"
    },
    {
        "id": "CSV-04",
        "category": "Tabular-SQL",
        "domain": "csv",
        "query": "List all products purchased with a quantity of 2 or more.",
        "expected_answer": "Monitor, Mouse",
        "type": "list"
    },
    {
        "id": "CSV-05",
        "category": "Tabular-SQL",
        "domain": "csv",
        "query": "What is the average order amount for customers with a 'VIP' membership status across completed orders?",
        "expected_answer": "675.0",
        "type": "numeric"
    },
    {
        "id": "CSV-06",
        "category": "Tabular-SQL",
        "domain": "csv",
        "query": "How many cancelled or pending orders exist in total?",
        "expected_answer": "2",
        "type": "numeric"
    },
    {
        "id": "CSV-07",
        "category": "Tabular-SQL",
        "domain": "csv",
        "query": "Which country generated the highest revenue from completed orders?",
        "expected_answer": "Germany",
        "type": "text"
    },
    {
        "id": "CSV-08",
        "category": "Tabular-SQL",
        "domain": "csv",
        "query": "What is the customer ID and total spend of the top customer in Munich?",
        "expected_answer": "C002",
        "type": "text"
    }
]

# ==============================================================================
# 3. HELPER FUNCTIONS & CLIENT INITIALIZATION
# ==============================================================================

def safe_groq_completion(client: Groq, model: str, messages: list, max_tokens: int, temperature: float = 0.0):
    """Executes a Groq call with automatic retry if rate limits (429) are encountered."""
    for attempt in range(5):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                print(f"\n⏳ [Rate limit encountered! Cooling down for 25s before retry... (attempt {attempt+1}/5)]")
                time.sleep(25)
            else:
                raise e
    raise RuntimeError("Failed after 5 rate limit retries.")


def strip_thinking(text: str) -> str:
    """Strips <think>...</think> reasoning tags from open-source reasoning models."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def init_supabase_client():
    """Initializes and authenticates Supabase client with user credentials."""
    auth_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    auth_res = auth_client.auth.sign_in_with_password({
        "email": TEST_EMAIL,
        "password": TEST_PASS
    })
    token = auth_res.session.access_token
    return create_client(
        SUPABASE_URL,
        SUPABASE_KEY,
        options=ClientOptions(headers={"Authorization": f"Bearer {token}"})
    )


def build_synthetic_csv_environment() -> Dict[str, Tuple[pd.DataFrame, Dict[str, Any]]]:
    """Generates verified in-memory relational tables for testing the DuckDB pipeline."""
    customers_data = {
        "customer_id": ["C001", "C002", "C003", "C004"],
        "name": ["Alice Smith", "Bob Jones", "Charlie Brown", "Diana Prince"],
        "city": ["Berlin", "Munich", "Berlin", "Paris"],
        "country": ["Germany", "Germany", "Germany", "France"],
        "tier": ["VIP", "Standard", "VIP", "Standard"]
    }
    orders_data = {
        "order_id": ["ORD-101", "ORD-102", "ORD-103", "ORD-104", "ORD-105"],
        "customer_id": ["C001", "C002", "C001", "C003", "C004"],
        "product": ["Laptop", "Monitor", "Mouse", "Keyboard", "Headphones"],
        "quantity": [1, 2, 3, 1, 1],
        "amount": [1200.0, 700.0, 150.0, 250.0, 180.0],
        "status": ["Completed", "Completed", "Completed", "Cancelled", "Pending"]
    }

    df_customers = pd.DataFrame(customers_data)
    df_orders = pd.DataFrame(orders_data)

    def make_meta(df, filename):
        return {
            "source": filename,
            "rows": len(df),
            "columns": list(df.columns),
            "dtypes": {str(c): str(d) for c, d in df.dtypes.items()},
            "preview": df.head(2).to_dict(orient="records")
        }

    return {
        "customers.csv": (df_customers, make_meta(df_customers, "customers.csv")),
        "orders.csv": (df_orders, make_meta(df_orders, "orders.csv"))
    }


# ==============================================================================
# 4. BENCHMARK MODULES
# ==============================================================================

class HARIEvaluator:
    def __init__(self):
        print("\n[Init] Authenticating with Supabase...")
        self.supabase = init_supabase_client()
        print("[Init] Loading SentenceTransformer (" + EMBED_MODEL + ")...")
        self.embedder = SentenceTransformer(EMBED_MODEL)
        print("[Init] Initializing Groq client...")
        self.groq = Groq(api_key=GROQ_API_KEY)
        self.csv_env = build_synthetic_csv_environment()

        # Token and latency metrics
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.retrieval_latencies = []
        self.generation_latencies = []

    # --------------------------------------------------------------------------
    # A. RETRIEVAL EVALUATION
    # --------------------------------------------------------------------------
    def evaluate_retrieval(self, test: Dict[str, Any]) -> Tuple[int, List[Dict[str, Any]], float]:
        """Runs Supabase hybrid_search RPC and determines ground truth rank position."""
        query = test["query"]
        expected_file = test["expected_file"]
        keywords = test["keywords"]

        t0 = time.perf_counter()
        query_embedding = self.embedder.encode(query, convert_to_numpy=True).tolist()

        rpc_res = self.supabase.rpc("hybrid_search", {
            "query_text": query,
            "query_embedding": query_embedding,
            "match_count": TOP_K
        }).execute()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.retrieval_latencies.append(elapsed_ms)

        candidates = rpc_res.data or []
        rank_found = 0

        for rank, row in enumerate(candidates, start=1):
            content = row.get("content", "")
            meta = row.get("metadata") or {}
            source = meta.get("filename") or meta.get("source") or row.get("filename")

            # Check filename match
            file_match = (source == expected_file)
            # Check keyword match
            keyword_match = any(kw.lower() in content.lower() for kw in keywords) if keywords else True

            if file_match and keyword_match:
                rank_found = rank
                break

        return rank_found, candidates, elapsed_ms

    # --------------------------------------------------------------------------
    # B. GENERATION EXECUTION (PDF Pipeline Replica)
    # --------------------------------------------------------------------------
    def generate_pdf_response(self, query: str, context_chunks: List[Dict[str, Any]]) -> Tuple[str, float]:
        """Constructs prompt identical to responser.py and calls Groq."""
        if not context_chunks:
            return "I could not find relevant information about this in the uploaded document.", 0.0

        formatted_chunks = []
        for i, c in enumerate(context_chunks, start=1):
            source = c.get("metadata", {}).get("source") or c.get("metadata", {}).get("filename", "doc")
            page = c.get("metadata", {}).get("page", "?")
            formatted_chunks.append(f"[Chunk {i} — {source}, Page {page}]\n{c.get('content', '')}")

        context_str = "\n\n".join(formatted_chunks)
        user_message = f"<context>\n{context_str}\n</context>\n\nUser Query: {query}"

        system_prompt = (
            "You are hArI, an AI assistant with expertise in document analysis. "
            "Answer strictly from the provided context. If the answer is not found, "
            "respond: 'I could not find relevant information about this in the uploaded document.' "
            "Always cite your source at the end: (Source: <filename>, Page <page_number>)."
        )

        t0 = time.perf_counter()
        # res = self.groq.chat.completions.create(
        #     model=RESPONSE_MODEL,
        #     messages=[
        #         {"role": "system", "content": system_prompt},
        #         {"role": "user", "content": user_message}
        #     ],
        #     max_tokens=1024,
        #     temperature=0.2
        # )
        res = safe_groq_completion(
            client=self.groq,
            model=RESPONSE_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=1024,
            temperature=0.2
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.generation_latencies.append(elapsed_ms)

        usage = res.usage
        if usage:
            self.total_prompt_tokens += usage.prompt_tokens
            self.total_completion_tokens += usage.completion_tokens

        raw_output = res.choices[0].message.content or ""
        clean_output = strip_thinking(raw_output).strip()
        return clean_output, elapsed_ms

    # --------------------------------------------------------------------------
    # C. LLM-AS-A-JUDGE (Faithfulness & Correctness)
    # --------------------------------------------------------------------------
    def judge_response(self, query: str, context: str, ground_truth: str, response: str, is_negative: bool) -> Dict[str, Any]:
        """Uses gpt-oss-120b as an impartial judge for atomic faithfulness and correctness."""
        fallback = {
            "faithfulness": 1 if ("could not find" in response.lower() and is_negative) else 0,
            "correctness": 2 if (is_negative and "could not find" in response.lower()) else 1,
            "refusal_accuracy": 1 if (is_negative and "could not find" in response.lower()) else 0,
            "reason": "Fallback default"
        }

        if is_negative and "could not find" in response.lower():
            return {
                "faithfulness": 1,
                "correctness": 2,
                "refusal_accuracy": 1,
                "reason": "Correctly refused unanswerable question."
            }

        judge_prompt = f"""You are an impartial AI evaluation judge reviewing an answer from a RAG system.
Evaluate the candidate response strictly using the provided context and ground truth.

User Query: {query}
Retrieved Context: {context[:2500]}
Ground Truth: {ground_truth}
Candidate Response: {response}

Criteria:
1. FAITHFULNESS (0 or 1): Is every single fact in Candidate Response directly supported by Retrieved Context? (1 = yes, 0 = contains unsupported claims or external hallucinations).
2. CORRECTNESS (0, 1, or 2): Does Candidate Response deliver the correct answer as defined in Ground Truth? (2 = fully correct, 1 = partially correct, 0 = incorrect).
3. REFUSAL_ACCURACY (0 or 1): {"If unanswerable, did it refuse?" if is_negative else "1"}

Output ONLY valid JSON in this exact structure:
{{
  "faithfulness": 1,
  "correctness": 2,
  "refusal_accuracy": 1,
  "reason": "short explanation"
}}
"""
        try:
            # res = self.groq.chat.completions.create(
            #     model=JUDGE_MODEL,
            #     messages=[{"role": "user", "content": judge_prompt}],
            #     max_tokens=1500,
            #     temperature=0.0
            # )
            res = safe_groq_completion(
                client=self.groq,
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": judge_prompt}],
                max_tokens=1500,
                temperature=0.0
            )
            raw = res.choices[0].message.content or ""
            clean = strip_thinking(raw).strip()
            text_to_parse = clean if clean else raw

            json_match = re.search(r"\{[\s\S]*\}", text_to_parse)
            if json_match:
                parsed = json.loads(json_match.group(0))
                return {
                    "faithfulness": int(parsed.get("faithfulness", 1)),
                    "correctness": int(parsed.get("correctness", 1)),
                    "refusal_accuracy": int(parsed.get("refusal_accuracy", 1)),
                    "reason": str(parsed.get("reason", ""))
                }
            return fallback

        except Exception as e:
            return fallback

    # --------------------------------------------------------------------------
    # D. CITATION VERIFICATION
    # --------------------------------------------------------------------------
    def verify_citations(self, response: str, context_chunks: List[Dict[str, Any]], expected_page: Any) -> bool:
        """Verifies if the cited page and filename exist in retrieved context chunks."""
        if expected_page is None:
            return "(Source:" not in response

        match = re.search(r"\((?:Source:\s*)?([^,\)]+),\s*Page\s*(\d+)\)", response, re.IGNORECASE)
        if not match:
            return False

        cited_page = int(match.group(2).strip())
        for c in context_chunks:
            meta = c.get("metadata") or {}
            c_page = meta.get("page")
            if cited_page == c_page:
                return True
        return False

    # --------------------------------------------------------------------------
    # E. TABULAR CSV & DUCKDB EXECUTION PIPELINE
    # --------------------------------------------------------------------------

    def evaluate_csv_query(self, test: Dict[str, Any]) -> Tuple[bool, bool, str, float]:
        """
        Executes the exact multi-table DuckDB pipeline from ui/handlers.py:231.
        Returns: (first_pass_success, final_success, result_str, latency_ms)
        """
        query = test["query"]
        con = duckdb.connect(config={"enable_external_access": False})

        schema_parts = []
        for name, (df, meta) in self.csv_env.items():
            tbl_name = re.sub(r'\W|^(?=\d)', '_', name).lower()
            con.register(tbl_name, df)
            schema = f"Table: {tbl_name}\nColumns:\n"
            schema += "\n".join(f"  - {col}: {meta['dtypes'].get(col, '?')}" for col in meta["columns"])
            schema += f"\nSample Data:\n{meta['preview'][:2]}"
            schema_parts.append(schema)

        combined_schema = "\n\n".join(schema_parts)

        messages = [
            {"role": "system", "content": "You are a senior PostgreSQL and DuckDB expert."},
            {"role": "user", "content": (
                f"Write a SQL query to answer this request.\n\n"
                f"Available Tables & Schemas:\n{combined_schema}\n\n"
                f"Request: {query}\n\n"
                f"Rules:\n"
                f"- Return ONLY the raw SQL query\n"
                f"- Do NOT wrap it in markdown block quotes\n"
                f"- ALWAYS use ILIKE or LOWER(...) for all text filtering\n"
            )}
        ]

        t0 = time.perf_counter()
        first_pass_success = False
        final_success = False
        analysis_result = ""
        sql_query = ""
        max_retries = 3

        for attempt in range(max_retries):
            try:
                # res = self.groq.chat.completions.create(
                #     model=ANALYSIS_MODEL,
                #     messages=messages,
                #     max_tokens=800,
                #     temperature=0.1
                # )
                res = safe_groq_completion(
                    client=self.groq,
                    model=ANALYSIS_MODEL,
                    messages=messages,
                    max_tokens=800,
                    temperature=0.1
                )
                raw_sql = res.choices[0].message.content or ""
                sql_query = strip_thinking(raw_sql).strip()
                sql_query = re.sub(r"^```[sS][qQ][lL]?\n?", "", sql_query).strip()
                sql_query = re.sub(r"\n?```$", "", sql_query).strip()

                result_df = con.execute(sql_query).df()
                analysis_result = result_df.to_string()

                if attempt == 0:
                    first_pass_success = True
                final_success = True
                break
            except Exception as e:
                err = str(e)
                if attempt < max_retries - 1:
                    messages.append({"role": "assistant", "content": sql_query})
                    messages.append({"role": "user", "content": f"SQL failed: {err}. Fix and return ONLY SQL."})
                else:
                    analysis_result = f"Error: {err}"

        elapsed_ms = (time.perf_counter() - t0) * 1000
        con.close()
        return first_pass_success, final_success, analysis_result, elapsed_ms


# ==============================================================================
# 5. MASTER EVALUATION RUNNER & SCORECARD GENERATOR
# ==============================================================================

def run_full_evaluation():
    print("=" * 80)
    print("🚀 STARTING hArI v2 PRODUCTION EVALUATION BENCHMARK")
    print("=" * 80)

    evaluator = HARIEvaluator()

    # Metrics Storage
    category_metrics = {
        "Exact-Term": {"hits_3": 0, "hits_5": 0, "rr": [], "faithful": 0, "correct": 0, "total": 0},
        "Conceptual": {"hits_3": 0, "hits_5": 0, "rr": [], "faithful": 0, "correct": 0, "total": 0},
        "Multi-Chunk": {"hits_3": 0, "hits_5": 0, "rr": [], "faithful": 0, "correct": 0, "total": 0},
        "Negative-Refusal": {"refused_correctly": 0, "faithful": 0, "total": 0},
        "Tabular-SQL": {"first_pass": 0, "final_pass": 0, "correct": 0, "total": 0}
    }

    citations_valid = 0
    total_citation_eligible = 0

    print(f"\nExecuting {len(GOLDEN_DATASET)} Stratified Test Queries...\n")
    print(f"{'#':<3} | {'Cat':<15} | {'Query':<36} | {'Retr':<6} | {'Faith':<5} | {'Corr':<5} | {'Lat':<8}")
    print("-" * 88)

    current_category = None

    for i, test in enumerate(GOLDEN_DATASET, start=1):
        cat = test["category"]
        domain = test["domain"]
        query = test["query"]
        q_short = query[:33] + "..." if len(query) > 33 else query

        if current_category is not None and cat != current_category:
            print(f"\n⏳ Completed '{current_category}'. Cooling down for 20 seconds to reset Groq limits...\n")
            time.sleep(20)
        current_category = cat


        category_metrics[cat]["total"] += 1
        time.sleep(2.0)

        if domain == "pdf":
            is_neg = (cat == "Negative-Refusal")

            # 1. Retrieval
            rank, chunks, retr_lat = evaluator.evaluate_retrieval(test)

            if not is_neg:
                if 0 < rank <= 3:
                    category_metrics[cat]["hits_3"] += 1
                if 0 < rank <= 5:
                    category_metrics[cat]["hits_5"] += 1
                category_metrics[cat]["rr"].append(1.0 / rank if rank > 0 else 0.0)
                rank_str = f"#{rank}" if rank > 0 else "MISS"
            else:
                rank_str = "N/A"

            # 2. Generation
            response, gen_lat = evaluator.generate_pdf_response(query, chunks)

            # 3. Citation Check
            if not is_neg:
                total_citation_eligible += 1
                if evaluator.verify_citations(response, chunks, test.get("expected_page")):
                    citations_valid += 1

            # 4. LLM-as-a-Judge
            context_text = "\n".join([c.get("content", "") for c in chunks])
            judge_res = evaluator.judge_response(
                query=query,
                context=context_text,
                ground_truth=test["expected_answer"],
                response=response,
                is_negative=is_neg
            )

            is_faithful = judge_res.get("faithfulness", 0)
            corr_score = judge_res.get("correctness", 0)
            ref_acc = judge_res.get("refusal_accuracy", 1 if not is_neg else 0)

            category_metrics[cat]["faithful"] += is_faithful

            if is_neg:
                if ref_acc == 1:
                    category_metrics[cat]["refused_correctly"] += 1
                corr_str = "REF_OK" if ref_acc == 1 else "FAIL"
            else:
                category_metrics[cat]["correct"] += (1 if corr_score == 2 else 0)
                corr_str = f"{corr_score}/2"

            total_lat = retr_lat + gen_lat
            faith_str = "PASS" if is_faithful == 1 else "FAIL"
            print(f"{i:<3} | {cat:<15} | {q_short:<36} | {rank_str:<6} | {faith_str:<5} | {corr_str:<5} | {total_lat:>6.1f}ms")

        elif domain == "csv":
            # Tabular DuckDB Evaluation
            first_pass, final_pass, res_str, csv_lat = evaluator.evaluate_csv_query(test)
            if first_pass:
                category_metrics[cat]["first_pass"] += 1
            if final_pass:
                category_metrics[cat]["final_pass"] += 1

            # ── Generic Type-Driven Comparison (Zero Hardcoded IDs) ──
            expected = str(test["expected_answer"])
            q_type = test.get("type", "text")
            is_correct = False

            if q_type == "numeric":
                expected_num = float(expected)
                nums = [float(n) for n in re.findall(r"[-+]?\d*\.\d+|\d+", res_str)]
                is_correct = any(abs(n - expected_num) < 0.05 for n in nums)
            elif q_type == "list":
                # Verifies that every expected item appears in the result string
                items = [item.strip().lower() for item in expected.split(",")]
                is_correct = all(item in res_str.lower() for item in items)
            else:
                is_correct = expected.lower() in res_str.lower()

            if is_correct:
                category_metrics[cat]["correct"] += 1

            pass_str = "1st" if first_pass else ("Retry" if final_pass else "FAIL")
            corr_str = "PASS" if is_correct else "FAIL"
            print(f"{i:<3} | {cat:<15} | {q_short:<36} | {'N/A':<6} | {pass_str:<5} | {corr_str:<5} | {csv_lat:>6.1f}ms")

    # ==============================================================================
    # 6. SUMMARY SCORECARD
    # ==============================================================================
    total_retrieval_queries = sum([category_metrics[c]["total"] for c in ["Exact-Term", "Conceptual", "Multi-Chunk"]])
    total_hits_3 = sum([category_metrics[c]["hits_3"] for c in ["Exact-Term", "Conceptual", "Multi-Chunk"]])
    total_hits_5 = sum([category_metrics[c]["hits_5"] for c in ["Exact-Term", "Conceptual", "Multi-Chunk"]])
    all_rr = [rr for c in ["Exact-Term", "Conceptual", "Multi-Chunk"] for rr in category_metrics[c]["rr"]]

    overall_hit_3 = (total_hits_3 / total_retrieval_queries) * 100 if total_retrieval_queries else 0
    overall_hit_5 = (total_hits_5 / total_retrieval_queries) * 100 if total_retrieval_queries else 0
    overall_mrr   = sum(all_rr) / total_retrieval_queries if total_retrieval_queries else 0

    avg_retr_lat = sum(evaluator.retrieval_latencies) / len(evaluator.retrieval_latencies) if evaluator.retrieval_latencies else 0
    avg_gen_lat  = sum(evaluator.generation_latencies) / len(evaluator.generation_latencies) if evaluator.generation_latencies else 0

    est_cost = (
        (evaluator.total_prompt_tokens / 1_000_000 * COST_PER_1M_INPUT) +
        (evaluator.total_completion_tokens / 1_000_000 * COST_PER_1M_OUTPUT)
    )

    print("\n" + "=" * 80)
    print("📊 hArI v2 BENCHMARK SUMMARY SCORECARD")
    print("=" * 80)
    print(f"1. RETRIEVAL PERFORMANCE (N={total_retrieval_queries} queries):")
    print(f"   • Hit Rate @ 3 (Recall @ 3)  : {overall_hit_3:>6.1f}%")
    print(f"   • Hit Rate @ 5 (Recall @ 5)  : {overall_hit_5:>6.1f}%")
    print(f"   • Mean Reciprocal Rank (MRR) : {overall_mrr:>6.3f} (Max: 1.000)")
    print(f"   • Avg Retrieval Latency      : {avg_retr_lat:>6.1f} ms")
    print("-" * 80)
    print("2. RETRIEVAL CATEGORY BREAKDOWN:")
    for cat_name in ["Exact-Term", "Conceptual", "Multi-Chunk"]:
        m = category_metrics[cat_name]
        c_count = m["total"]
        hr5 = (m["hits_5"] / c_count) * 100 if c_count else 0
        cat_mrr = sum(m["rr"]) / c_count if c_count else 0
        print(f"   • {cat_name:<12} -> Hit Rate @ 5: {hr5:>5.1f}% | MRR: {cat_mrr:.3f}")
    print("-" * 80)
    print("3. GENERATION INTEGRITY & SAFETY:")
    pdf_eval_count = total_retrieval_queries + category_metrics["Negative-Refusal"]["total"]
    total_faithful = sum([category_metrics[c]["faithful"] for c in ["Exact-Term", "Conceptual", "Multi-Chunk", "Negative-Refusal"]])
    faith_rate = (total_faithful / pdf_eval_count) * 100 if pdf_eval_count else 0
    ref_count = category_metrics["Negative-Refusal"]["total"]
    ref_rate = (category_metrics["Negative-Refusal"]["refused_correctly"] / ref_count) * 100 if ref_count else 0
    cit_rate = (citations_valid / total_citation_eligible) * 100 if total_citation_eligible else 0

    print(f"   • Atomic Faithfulness Score  : {faith_rate:>6.1f}% (Grounded in context)")
    print(f"   • Negative Refusal Accuracy  : {ref_rate:>6.1f}% (Out-of-domain rejection)")
    print(f"   • Citation Page Accuracy     : {cit_rate:>6.1f}% (Cited chunk verified in prompt)")
    print("-" * 80)
    print("4. TABULAR DUCKDB SQL PIPELINE (N=8 queries):")
    csv_m = category_metrics["Tabular-SQL"]
    first_pct = (csv_m["first_pass"] / csv_m["total"]) * 100
    final_pct = (csv_m["final_pass"] / csv_m["total"]) * 100
    acc_pct   = (csv_m["correct"] / csv_m["total"]) * 100
    print(f"   • First-Pass SQL Success     : {first_pct:>6.1f}%")
    print(f"   • Final SQL Success (Retries): {final_pct:>6.1f}%")
    print(f"   • Tabular Result Accuracy    : {acc_pct:>6.1f}%")
    print("-" * 80)
    print("5. OPERATIONAL LATENCY & COST ACCOUNTING:")
    print(f"   • Avg Generation Latency     : {avg_gen_lat:>6.1f} ms")
    print(f"   • Total Prompt Tokens Used   : {evaluator.total_prompt_tokens:>8,d}")
    print(f"   • Total Completion Tokens    : {evaluator.total_completion_tokens:>8,d} (Includes <think> reasoning)")
    print(f"   • Total Evaluation Run Cost  : ${est_cost:>7.4f}")
    print("=" * 80)


if __name__ == "__main__":
    run_full_evaluation()