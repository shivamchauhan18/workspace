"""
RAG Evaluation Script for RAG Pipeline
=====================================
Evaluates the RAG pipeline using multiple approaches:

1. RAGAS Metrics:
   - Context Precision
   - Context Recall
   - Faithfulness

2. LLM-as-Judge Accuracy:
   - Binary correct/wrong judgment
   - Confidence score (0.0-1.0)
   - Explanation for each judgment
   - Overall accuracy percentage

3. Simple Word-Overlap Metrics (fallback):
   - Word recall, precision, F1

Uses local Ollama LLM + embeddings (no OpenAI APIs).

Usage:
    # Full evaluation (RAGAS + LLM-Judge)
    python evaluation.py --llm-judge

    # Only LLM-as-Judge evaluation
    python evaluation.py --llm-judge-only

    # Quick test with limited samples
    python evaluation.py --max-samples 10 --llm-judge

    # Use cached pipeline results
    python evaluation.py --use-cached --llm-judge-only

Prerequisites:
    pip install ragas datasets
    Ollama must be running with the LLM and embedding models pulled.

    For reliable evaluation, pull a larger judge model:
    ollama pull qwen2.5:7b    # Recommended for best results
    ollama pull llama3.1:8b  # Alternative
"""

import json
import sys
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so config/rag_pipeline are importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from rag_pipeline import (
    create_vector_store,
    create_non_llm_documents,
    retrieve_context,
    generate_answer,
    get_qdrant_client,
    collection_exists,
    documents_changed,
)

# ---------------------------------------------------------------------------
# RAGAS imports
# ---------------------------------------------------------------------------
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)
from langchain_ollama import ChatOllama

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("RAGAS_EVAL")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Path to the QA dataset JSON file (one JSON object per line)
QA_DATASET_PATH = PROJECT_ROOT / "85_qa.json"

# IMPORTANT: RAGAS evaluation requires a capable LLM that can follow
# structured output instructions. Small models (< 3B params) typically fail
# to produce valid JSON outputs for RAGAS prompts.
#
# Recommended judge models (pull with ollama pull <model>):
#   - qwen2.5:7b (best balance of quality and speed)
#   - llama3.1:8b (good alternative)
#   - mistral:7b (another option)
#   - gemma3:4b (minimum viable, may still have issues)
#
# The pipeline's answer generation can still use the small model,
# but the JUDGE must be larger for reliable evaluation.

# Auto-detect available larger models, fallback to config default
EVAL_LLM_MODEL = config.OLLAMA_LLM_MODEL  # Default, may be too small
EVAL_EMBEDDING_MODEL = config.DENSE_EMBEDDING_MODEL
OLLAMA_BASE_URL = config.BASE_URL

# Recommended judge models in order of preference
RECOMMENDED_JUDGE_MODELS = [
    "qwen2.5:7b",
    "llama3.1:8b",
    "mistral:7b",
    "qwen2.5:3b",
    "gemma3:4b",
]

# ===========================================================================
# LLM-as-Judge Prompts for Answer Correctness Evaluation
# ===========================================================================

LLM_JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system. Your task is to judge whether the generated answer is correct based on the ground truth answer.

You must respond with EXACTLY one of these two words:
- "CORRECT" if the generated answer is factually accurate and aligns with the ground truth
- "WRONG" if the generated answer is factually incorrect, incomplete, or contradicts the ground truth

Guidelines for evaluation:
1. Focus on factual correctness, not style or wording
2. Minor differences in phrasing are acceptable if the meaning is preserved
3. Additional relevant information in the generated answer is acceptable
4. The generated answer is WRONG if it:
   - Contains factually incorrect information
   - Misses key information from the ground truth
   - Contradicts the ground truth
   - Is completely irrelevant to the question

Respond with ONLY "CORRECT" or "WRONG" - no explanations or additional text."""

LLM_JUDGE_USER_PROMPT = """Evaluate the following RAG system answer:

Question: {question}

Ground Truth Answer: {ground_truth}

Generated Answer: {generated_answer}

Is the generated answer CORRECT or WRONG?"""

LLM_JUDGE_DETAILED_PROMPT = """You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.

Your task is to evaluate the generated answer against the ground truth answer and provide:
1. A correctness judgment (CORRECT or WRONG)
2. A confidence score (0.0 to 1.0)
3. A brief explanation

Question: {question}

Ground Truth Answer: {ground_truth}

Generated Answer: {generated_answer}

Respond in the following JSON format ONLY:
{{
    "judgment": "CORRECT" or "WRONG",
    "confidence": <float between 0.0 and 1.0>,
    "explanation": "<brief explanation of your judgment>"
}}"""


def check_ollama_model_available(model_name: str) -> bool:
    """Check if an Ollama model is available locally."""
    import subprocess
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=30
        )
        return model_name in result.stdout
    except Exception:
        return False


def get_best_judge_model() -> str:
    """
    Find the best available model for RAGAS judging.
    Returns the first available model from the recommended list,
    or falls back to config.OLLAMA_LLM_MODEL if none found.
    """
    for model in RECOMMENDED_JUDGE_MODELS:
        if check_ollama_model_available(model):
            logger.info(f"Found recommended judge model: {model}")
            return model

    # Check if the config model is large enough
    config_model = config.OLLAMA_LLM_MODEL
    if check_ollama_model_available(config_model):
        # Warn if using a small model
        model_size = config_model.split(":")[-1] if ":" in config_model else ""
        if any(small in model_size for small in ["270m", "500m", "1b", "0.6b", "0.8b"]):
            logger.warning(
                f"Model {config_model} may be too small for reliable RAGAS evaluation. "
                f"Consider pulling a larger model: ollama pull qwen2.5:7b"
            )
        return config_model

    logger.warning("No suitable judge model found. Please pull one:")
    logger.warning("  ollama pull qwen2.5:7b")
    return config.OLLAMA_LLM_MODEL


# ===========================================================================
# Step 1: Load QA dataset
# ===========================================================================
def load_qa_dataset(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL-style QA dataset (one JSON object per line)."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                samples.append(obj)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed line {line_num}: {e}")
    logger.info(f"Loaded {len(samples)} QA samples from {path}")
    return samples


# ===========================================================================
# Step 2: Initialize RAG pipeline (vector store)
# ===========================================================================
def init_rag_pipeline() -> Any:
    """
    Initialize or load the vector store.
    Mirrors the logic in app.py / rag_pipeline.__main__.
    """
    logger.info("Initializing RAG pipeline...")

    client = get_qdrant_client()
    exists = collection_exists(client)

    if not exists:
        logger.info("Collection does not exist — creating from documents")
        docs = create_non_llm_documents()
        if not docs:
            raise RuntimeError("No documents found for vector store creation")
        store = create_vector_store(docs)
    else:
        # Check for changed files and ingest any new ones
        changed_files, current_hashes = documents_changed()
        if changed_files:
            logger.info(f"Detected {len(changed_files)} changed files — re-ingesting")
            docs = create_non_llm_documents(changed_files)
            store = create_vector_store(docs)
        else:
            logger.info("No changes detected — loading existing vector store")
            store = create_vector_store([])

    logger.info("RAG pipeline ready")
    return store


# ===========================================================================
# Step 3: Run pipeline for each question and collect contexts + answers
# ===========================================================================
def run_pipeline_on_dataset(
    store: Any,
    qa_samples: List[Dict[str, Any]],
    max_samples: Optional[int] = None,
) -> Dict[str, List]:
    """
    For each question in the dataset:
      - Retrieve contexts from the RAG pipeline
      - Generate an answer using the pipeline's LLM
      - Collect: question, answer, contexts (list of strings), ground_truth

    Returns a dict compatible with `Dataset.from_dict()`.
    """
    questions = []
    answers = []
    contexts_list = []
    ground_truths = []

    samples_to_process = qa_samples[:max_samples] if max_samples else qa_samples

    for i, sample in enumerate(samples_to_process):
        question = sample["question"]
        ground_truth = sample["answer"]

        logger.info(f"[{i+1}/{len(samples_to_process)}] Querying: {question[:80]}...")

        try:
            # Retrieve relevant documents
            retrieved_docs = retrieve_context(question, store)

            # Extract context strings
            ctx = [doc.page_content for doc in retrieved_docs]
            if not ctx:
                logger.warning(f"No contexts retrieved for question {i+1}")
                ctx = [""]

            # Generate answer
            answer, _ = generate_answer(store, question)

        except Exception as e:
            logger.error(f"Error on question {i+1}: {e}")
            ctx = [""]
            answer = ""

        questions.append(question)
        answers.append(answer)
        contexts_list.append(ctx)
        ground_truths.append(ground_truth)

        # Small delay to avoid overwhelming Ollama
        time.sleep(0.3)

    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    }


# ===========================================================================
# Step 4: RAGAS evaluate
# ===========================================================================
def run_ragas_evaluation(data: Dict[str, List], judge_model: str) -> Optional[Dict]:
    """
    Run RAGAS evaluation with ContextPrecision, ContextRecall, Faithfulness
    using local Ollama models.

    Returns None if evaluation fails.
    """
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.run_config import RunConfig

    logger.info("Setting up RAGAS evaluator with local Ollama models...")
    logger.info(f"  Judge model: {judge_model}")
    logger.info(f"  Embedding model: {EVAL_EMBEDDING_MODEL}")

    # Wrap Ollama LLM for RAGAS
    # Use larger context window for evaluation prompts
    eval_llm = ChatOllama(
        model=judge_model,
        temperature=0,
        base_url=OLLAMA_BASE_URL,
        num_ctx=8192,
        num_predict=2048,  # Allow longer responses for structured output
    )
    ragas_llm = LangchainLLMWrapper(eval_llm)

    # Wrap Ollama embeddings for RAGAS
    eval_embeddings = OllamaEmbeddings(
        model=EVAL_EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(eval_embeddings)

    # Create dataset
    dataset = Dataset.from_dict(data)
    logger.info(f"Evaluation dataset has {len(dataset)} samples")

    # RunConfig — generous timeouts for local inference
    run_config = RunConfig(
        timeout=300,       # 5 min per metric call (local models are slow)
        max_retries=5,     # Reduced retries, fail fast on format errors
        max_wait=120,
        max_workers=1,     # Serialize to avoid overwhelming local Ollama
    )

    # Define metrics - each needs the LLM
    metrics = [
        ContextPrecision(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
        Faithfulness(llm=ragas_llm),
    ]

    logger.info("Starting RAGAS evaluation (this may take a while with local models)...")
    logger.info(f"  Metrics: ContextPrecision, ContextRecall, Faithfulness")

    try:
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            run_config=run_config,
            raise_exceptions=False,  # Don't raise, return NaN for failed metrics
        )
        return result
    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        logger.error("This often happens with small models that can't follow JSON format instructions.")
        logger.error(f"Try: ollama pull qwen2.5:7b")
        return None


# ===========================================================================
# Alternative: Simple keyword-based metrics (fallback)
# ===========================================================================
def compute_simple_metrics(data: Dict[str, List]) -> Dict[str, List[float]]:
    """
    Compute simple fallback metrics when RAGAS evaluation fails.
    These don't require an LLM judge.
    """
    from collections import Counter
    import re

    def tokenize(text: str) -> set:
        """Simple tokenization."""
        return set(re.findall(r'\b\w+\b', text.lower()))

    def compute_recall(answer: str, ground_truth: str) -> float:
        """Compute word overlap recall."""
        ans_tokens = tokenize(answer)
        gt_tokens = tokenize(ground_truth)
        if not gt_tokens:
            return 0.0
        overlap = ans_tokens & gt_tokens
        return len(overlap) / len(gt_tokens)

    def compute_precision(answer: str, ground_truth: str) -> float:
        """Compute word overlap precision."""
        ans_tokens = tokenize(answer)
        gt_tokens = tokenize(ground_truth)
        if not ans_tokens:
            return 0.0
        overlap = ans_tokens & gt_tokens
        return len(overlap) / len(ans_tokens)

    def compute_f1(answer: str, ground_truth: str) -> float:
        """Compute F1 score."""
        p = compute_precision(answer, ground_truth)
        r = compute_recall(answer, ground_truth)
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    results = {
        "word_recall": [],
        "word_precision": [],
        "word_f1": [],
        "context_length": [],
    }

    for q, a, ctxs, gt in zip(
        data["question"],
        data["answer"],
        data["contexts"],
        data["ground_truth"]
    ):
        results["word_recall"].append(compute_recall(a, gt))
        results["word_precision"].append(compute_precision(a, gt))
        results["word_f1"].append(compute_f1(a, gt))
        results["context_length"].append(sum(len(c) for c in ctxs))

    return results


# ===========================================================================
# LLM-as-Judge: Answer Correctness Evaluation
# ===========================================================================
def llm_as_judge_evaluation(
    data: Dict[str, List],
    judge_model: str,
    detailed: bool = True,
) -> Dict[str, Any]:
    """
    Use LLM as a judge to evaluate answer correctness against ground truth.

    Args:
        data: Dictionary with 'question', 'answer', 'ground_truth' keys
        judge_model: Ollama model to use as judge
        detailed: If True, get detailed judgment with confidence and explanation

    Returns:
        Dictionary with 'judgments', 'confidences', 'explanations', 'accuracy'
    """
    logger.info(f"Starting LLM-as-Judge evaluation with model: {judge_model}")

    # Initialize judge LLM
    judge_llm = ChatOllama(
        model=judge_model,
        temperature=0,
        base_url=OLLAMA_BASE_URL,
        num_ctx=4096,
        num_predict=256,
    )

    judgments = []
    confidences = []
    explanations = []
    raw_responses = []

    total = len(data["question"])

    for i, (question, answer, ground_truth) in enumerate(zip(
        data["question"],
        data["answer"],
        data["ground_truth"]
    )):
        logger.info(f"Judging answer {i+1}/{total}...")

        # Skip if answer is empty
        if not answer or not answer.strip():
            judgments.append("WRONG")
            confidences.append(1.0)
            explanations.append("Empty answer")
            raw_responses.append("")
            continue

        try:
            if detailed:
                # Use detailed prompt for full evaluation
                prompt = LLM_JUDGE_DETAILED_PROMPT.format(
                    question=question,
                    ground_truth=ground_truth,
                    generated_answer=answer
                )

                response = judge_llm.invoke(prompt)
                raw_text = response.content.strip()
                raw_responses.append(raw_text)

                # Parse JSON response
                try:
                    # Handle potential markdown code blocks
                    raw_text = raw_text.strip()
                    if raw_text.startswith("```"):
                        raw_text = raw_text.split("```")[1]
                        if raw_text.startswith("json"):
                            raw_text = raw_text[4:]
                        raw_text = raw_text.strip()

                    result = json.loads(raw_text)
                    judgment = result.get("judgment", "WRONG").upper()
                    confidence = float(result.get("confidence", 0.5))
                    explanation = result.get("explanation", "")
                except (json.JSONDecodeError, ValueError):
                    # Fallback: parse text response
                    if "CORRECT" in raw_text.upper():
                        judgment = "CORRECT"
                    else:
                        judgment = "WRONG"
                    confidence = 0.5
                    explanation = f"Failed to parse: {raw_text[:100]}"
            else:
                # Use simple binary prompt
                messages = [
                    ("system", LLM_JUDGE_SYSTEM_PROMPT),
                    ("human", LLM_JUDGE_USER_PROMPT.format(
                        question=question,
                        ground_truth=ground_truth,
                        generated_answer=answer
                    ))
                ]

                response = judge_llm.invoke(messages)
                raw_text = response.content.strip().upper()
                raw_responses.append(raw_text)

                if "CORRECT" in raw_text:
                    judgment = "CORRECT"
                else:
                    judgment = "WRONG"
                confidence = 1.0 if judgment in raw_text else 0.5
                explanation = ""

            judgments.append(judgment)
            confidences.append(confidence)
            explanations.append(explanation)

        except Exception as e:
            logger.error(f"Error judging answer {i+1}: {e}")
            judgments.append("ERROR")
            confidences.append(0.0)
            explanations.append(str(e))
            raw_responses.append("")

        # Small delay to avoid overwhelming Ollama
        time.sleep(0.2)

    # Calculate accuracy metrics
    correct_count = sum(1 for j in judgments if j == "CORRECT")
    total_valid = sum(1 for j in judgments if j in ["CORRECT", "WRONG"])

    accuracy = correct_count / total_valid if total_valid > 0 else 0.0
    weighted_accuracy = sum(
        c for j, c in zip(judgments, confidences) if j == "CORRECT"
    ) / total_valid if total_valid > 0 else 0.0

    results = {
        "judgments": judgments,
        "confidences": confidences,
        "explanations": explanations,
        "raw_responses": raw_responses,
        "accuracy": accuracy,
        "weighted_accuracy": weighted_accuracy,
        "correct_count": correct_count,
        "total_evaluated": total_valid,
        "error_count": sum(1 for j in judgments if j == "ERROR"),
    }

    logger.info(f"LLM-Judge Accuracy: {accuracy:.2%} ({correct_count}/{total_valid})")

    return results


def save_llm_judge_results(
    data: Dict[str, List],
    judge_results: Dict[str, Any],
    output_path: Path,
) -> None:
    """Save LLM-as-judge evaluation results to JSON file."""
    results_list = []

    for i in range(len(data["question"])):
        results_list.append({
            "question": data["question"][i],
            "ground_truth": data["ground_truth"][i],
            "generated_answer": data["answer"][i],
            "judgment": judge_results["judgments"][i],
            "confidence": judge_results["confidences"][i],
            "explanation": judge_results["explanations"][i],
        })

    output = {
        "summary": {
            "accuracy": judge_results["accuracy"],
            "weighted_accuracy": judge_results["weighted_accuracy"],
            "correct_count": judge_results["correct_count"],
            "total_evaluated": judge_results["total_evaluated"],
            "error_count": judge_results["error_count"],
        },
        "per_question_results": results_list,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"LLM-Judge results saved to {output_path}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(description="RAGAS Evaluation for RAG Pipeline")
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Ollama model to use as judge (e.g., qwen2.5:7b). Auto-detected if not specified."
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for quick testing)."
    )
    parser.add_argument(
        "--use-cached",
        action="store_true",
        help="Use cached pipeline results if available (skip RAG pipeline run)."
    )
    parser.add_argument(
        "--simple-metrics",
        action="store_true",
        help="Only compute simple word-overlap metrics (no LLM needed)."
    )
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="Run LLM-as-Judge evaluation for answer correctness."
    )
    parser.add_argument(
        "--llm-judge-only",
        action="store_true",
        help="Run only LLM-as-Judge evaluation (skip RAGAS)."
    )
    parser.add_argument(
        "--detailed-judge",
        action="store_true",
        default=True,
        help="Get detailed judgments with confidence and explanation (default: True)."
    )
    args = parser.parse_args()

    # Determine judge model
    if args.judge_model:
        judge_model = args.judge_model
    else:
        judge_model = get_best_judge_model()

    # Step 1: Load QA dataset
    qa_samples = load_qa_dataset(QA_DATASET_PATH)
    if not qa_samples:
        logger.error("No QA samples loaded — aborting")
        sys.exit(1)

    # Step 2: Get pipeline results
    intermediate_path = PROJECT_ROOT / "eval_pipeline_results.json"

    if args.use_cached and intermediate_path.exists():
        logger.info(f"Loading cached results from {intermediate_path}")
        with open(intermediate_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Ensure proper list format
        if isinstance(data, dict):
            data = {k: list(v) for k, v in data.items()}
    else:
        # Initialize RAG pipeline
        store = init_rag_pipeline()

        # Run pipeline on each question
        logger.info("Running RAG pipeline on all questions...")
        data = run_pipeline_on_dataset(store, qa_samples, max_samples=args.max_samples)

        # Save intermediate results
        with open(intermediate_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved pipeline results to {intermediate_path}")

    # Step 3: Compute metrics
    if args.simple_metrics:
        # Only compute simple metrics
        logger.info("Computing simple word-overlap metrics...")
        simple_results = compute_simple_metrics(data)

        print("\n" + "=" * 60)
        print("SIMPLE METRICS RESULTS (Word Overlap)")
        print("=" * 60)
        for metric_name, values in simple_results.items():
            avg = sum(values) / len(values) if values else 0
            print(f"  {metric_name}: {avg:.4f}")

        # Save simple results
        simple_path = PROJECT_ROOT / "eval_simple_results.json"
        simple_output = []
        for i in range(len(data["question"])):
            simple_output.append({
                "question": data["question"][i],
                "answer": data["answer"][i],
                "ground_truth": data["ground_truth"][i],
                **{k: v[i] for k, v in simple_results.items()}
            })
        with open(simple_path, "w", encoding="utf-8") as f:
            json.dump(simple_output, f, indent=2, ensure_ascii=False)
        logger.info(f"Simple results saved to {simple_path}")
        return

    # ===========================================================================
    # LLM-as-Judge Evaluation
    # ===========================================================================
    if args.llm_judge or args.llm_judge_only:
        logger.info("Running LLM-as-Judge evaluation...")

        judge_results = llm_as_judge_evaluation(
            data,
            judge_model,
            detailed=args.detailed_judge
        )

        # Save LLM-judge results
        judge_results_path = PROJECT_ROOT / "eval_llm_judge_results.json"
        save_llm_judge_results(data, judge_results, judge_results_path)

        # Print summary
        print("\n" + "=" * 60)
        print("LLM-AS-JUDGE EVALUATION RESULTS")
        print("=" * 60)
        print(f"\n  Overall Accuracy: {judge_results['accuracy']:.2%}")
        print(f"  Weighted Accuracy: {judge_results['weighted_accuracy']:.2%}")
        print(f"  Correct: {judge_results['correct_count']}/{judge_results['total_evaluated']}")
        print(f"  Errors: {judge_results['error_count']}")

        # Show per-question breakdown
        print("\n  Per-Question Judgments:")
        print("  " + "-" * 56)
        for i, (q, j, c, e) in enumerate(zip(
            data["question"],
            judge_results["judgments"],
            judge_results["confidences"],
            judge_results["explanations"]
        )):
            q_short = q[:50] + "..." if len(q) > 50 else q
            status = "✓" if j == "CORRECT" else "✗"
            print(f"  [{status}] Q{i+1}: {q_short}")
            if e:
                print(f"       Explanation: {e[:80]}{'...' if len(e) > 80 else ''}")

        print(f"\n  Detailed results saved to: {judge_results_path}")

        if args.llm_judge_only:
            print(f"\n  Pipeline outputs: {intermediate_path}")
            return

    # Step 4: RAGAS evaluation
    result = run_ragas_evaluation(data, judge_model)

    if result is None:
        logger.error("RAGAS evaluation failed. Falling back to simple metrics.")
        simple_results = compute_simple_metrics(data)

        print("\n" + "=" * 60)
        print("FALLBACK: SIMPLE METRICS (RAGAS Failed)")
        print("=" * 60)
        for metric_name, values in simple_results.items():
            avg = sum(values) / len(values) if values else 0
            print(f"  {metric_name}: {avg:.4f}")
        return

    # Print results
    logger.info("=" * 60)
    logger.info("RAGAS EVALUATION RESULTS")
    logger.info("=" * 60)
    print("\n" + "=" * 60)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 60)

    # result is a pandas DataFrame-like object
    print(result)

    # Save results
    results_path = PROJECT_ROOT / "eval_ragas_results.json"
    result_df = result.to_pandas()
    result_df.to_json(results_path, orient="records", indent=2)
    logger.info(f"Detailed per-sample results saved to {results_path}")

    # Print summary
    print("\n--- Metric Averages ---")
    for metric_name in ["context_precision", "context_recall", "faithfulness"]:
        if metric_name in result_df.columns:
            # Handle NaN values
            valid_values = result_df[metric_name].dropna()
            if len(valid_values) > 0:
                avg = valid_values.mean()
                print(f"  {metric_name}: {avg:.4f} ({len(valid_values)}/{len(result_df)} valid)")
            else:
                print(f"  {metric_name}: N/A (all values invalid)")

    # Also print LLM-judge results if available
    judge_results_path = PROJECT_ROOT / "eval_llm_judge_results.json"
    if judge_results_path.exists():
        with open(judge_results_path, "r", encoding="utf-8") as f:
            judge_data = json.load(f)
        summary = judge_data.get("summary", {})
        print("\n--- LLM-as-Judge Accuracy ---")
        print(f"  Overall Accuracy: {summary.get('accuracy', 0):.2%}")
        print(f"  Weighted Accuracy: {summary.get('weighted_accuracy', 0):.2%}")
        print(f"  Correct: {summary.get('correct_count', 0)}/{summary.get('total_evaluated', 0)}")

    print(f"\nPer-sample results: {results_path}")
    print(f"Pipeline outputs:  {intermediate_path}")


if __name__ == "__main__":
    main()
