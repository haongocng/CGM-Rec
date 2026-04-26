"""Phase 1-4 CLI for CGM-Rec foundations and online graph-memory updates."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import Phase1Config
from data.loader import DatasetLoader
from data.splitter import split_warmup
from engine.diagnostics import HybridDiagnostics
from engine.test_loop import evaluate_llm_reranker_online, evaluate_semantic_scorer, evaluate_semantic_scorer_online
from engine.train_loop import train_semantic_scorer
from graph.seed_builder import SeedGraphBuilder
from llm.config import load_project_env
from llm.lesson_agent import LLMLessonAgent
from llm.manager import LanguageModelManager
from llm.prompt_builder import Phase5PromptBuilder
from llm.reranker import LLMReranker
from memory.episodic_memory import EpisodicMemory
from memory.semantic_memory import SemanticMemory
from memory.write_policy import WritePolicy
from memory.writer import MemoryWriter, MemoryWriterConfig
from model.scorer import LinearSemanticScorer
from retrieval.llm_evidence_builder import LLMEvidenceBuilder
from retrieval.semantic_retriever import SemanticRetriever
from utils.io import write_json, write_text
from utils.seed import set_random_seed
from utils.text import compact_text, normalize_title


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CGM-Rec Phase 1-4 foundations: inspect data, build memory, train a scorer, and run online updates.",
    )
    parser.add_argument(
        "--view",
        default="data",
        choices=["data", "seed-graph", "phase3-train", "phase3-test", "phase4-test-online", "phase5-test-llm"],
        help="Inspection/runtime target.",
    )
    parser.add_argument(
        "--dataset",
        default="ml_100k",
        help="Dataset name under cgm/dataset/ (current implementation targets ml_100k).",
    )
    parser.add_argument(
        "--warmup-mode",
        default="ratio",
        choices=["ratio", "count"],
        help="How to define the chronological warm-up prefix.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.2,
        help="Warm-up ratio when --warmup-mode=ratio.",
    )
    parser.add_argument(
        "--warmup-count",
        type=int,
        default=50,
        help="Warm-up count when --warmup-mode=count.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for reproducibility setup.",
    )
    parser.add_argument(
        "--keyword-top-k",
        type=int,
        default=5,
        help="Maximum number of keywords per item to include in the seed graph.",
    )
    parser.add_argument(
        "--co-occur-window-size",
        type=int,
        default=5,
        help="Sliding window size used to aggregate warm-up co-occurrence edges.",
    )
    parser.add_argument(
        "--no-description",
        action="store_true",
        help="Disable description nodes in the seed graph.",
    )
    parser.add_argument(
        "--inspect-title",
        type=str,
        default="",
        help="Optional title to inspect in the seed graph summary.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Phase 3 training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Phase 3 SGD learning rate.",
    )
    parser.add_argument(
        "--model-json",
        type=str,
        default="",
        help="Optional path under cgm/ to save or load scorer weights as JSON.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional path under cgm/ to save the main summary as JSON.",
    )
    parser.add_argument(
        "--dump-graph-json",
        type=str,
        default="",
        help="Optional path under cgm/ to export the full seed graph as JSON (seed-graph view only).",
    )
    parser.add_argument(
        "--dump-graph-text",
        type=str,
        default="",
        help="Optional path under cgm/ to export the full seed graph as plain text (seed-graph view only).",
    )
    parser.add_argument(
        "--episodic-max-records",
        type=int,
        default=500,
        help="Maximum number of episode records retained during Phase 4 online evaluation.",
    )
    parser.add_argument(
        "--max-tentative-edges",
        type=int,
        default=250,
        help="Maximum number of tentative online edges retained in semantic memory.",
    )
    parser.add_argument(
        "--llm-provider",
        default="timelygpt",
        choices=["mock", "openai", "deepinfra", "timelygpt"],
        help="Phase 5 LLM provider. Use mock for local architecture tests.",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=20,
        help="Maximum test samples for Phase 5 LLM runs. Use 0 to run the full test set.",
    )
    return parser


def load_bundle_and_config(project_root: Path, args: argparse.Namespace):
    # Build Phase1Config with the correct dataset_name upfront so that
    # __post_init__ resolves train/test/info paths for the right dataset.
    config = Phase1Config(
        dataset_root=project_root / "dataset",
        dataset_name=args.dataset,
        warmup_mode=args.warmup_mode,
        warmup_ratio=args.warmup_ratio,
        warmup_count=args.warmup_count,
        random_seed=args.seed,
        keyword_top_k=args.keyword_top_k,
        include_description=not args.no_description,
        co_occur_window_size=args.co_occur_window_size,
        phase3_epochs=args.epochs,
        phase3_learning_rate=args.learning_rate,
    )

    set_random_seed(config.random_seed)

    loader = DatasetLoader(expected_candidate_count=20)
    bundle = loader.load_dataset(
        config.dataset_name,
        str(config.train_path),
        str(config.test_path),
        str(config.info_path),
    )
    return config, bundle


def build_summary(project_root: Path, args: argparse.Namespace) -> dict:
    config, bundle = load_bundle_and_config(project_root, args)
    warmup = split_warmup(
        bundle.train_samples,
        mode=config.warmup_mode,
        warmup_ratio=config.warmup_ratio,
        warmup_count=config.warmup_count,
    )

    train_sample = bundle.train_samples[0]
    test_sample = bundle.test_samples[0]
    product_example = bundle.products[normalize_title(train_sample.target)]

    return {
        "dataset_name": bundle.dataset_name,
        "paths": {
            "train_path": str(config.train_path),
            "test_path": str(config.test_path),
            "info_path": str(config.info_path),
        },
        "counts": {
            "train_samples": len(bundle.train_samples),
            "test_samples": len(bundle.test_samples),
            "products": len(bundle.products),
            "warmup_samples": len(warmup.warmup_samples),
            "continual_stream_samples": len(warmup.stream_samples),
        },
        "warmup": {
            "mode": config.warmup_mode,
            "ratio": config.warmup_ratio,
            "count": config.warmup_count,
            "chronological": True,
        },
        "examples": {
            "train_sample_id": train_sample.sample_id,
            "train_target": train_sample.target,
            "train_target_position": train_sample.target_position,
            "train_target_index": train_sample.target_index,
            "train_target_index_base": train_sample.target_index_base,
            "train_session_items": train_sample.parsed_input.session_items[:5],
            "train_candidate_items": train_sample.parsed_input.candidate_items[:5],
            "test_sample_id": test_sample.sample_id,
            "test_target": test_sample.target,
            "product_example": {
                "title": product_example.title,
                "taxonomy_levels": product_example.taxonomy_levels,
                "keywords": product_example.keywords[:5],
                "description_preview": compact_text(product_example.description, limit=160),
            },
        },
    }


def build_seed_graph(project_root: Path, args: argparse.Namespace):
    config, bundle = load_bundle_and_config(project_root, args)
    warmup = split_warmup(
        bundle.train_samples,
        mode=config.warmup_mode,
        warmup_ratio=config.warmup_ratio,
        warmup_count=config.warmup_count,
    )
    return SeedGraphBuilder(
        keyword_top_k=config.keyword_top_k,
        include_description=config.include_description,
        co_occur_window_size=config.co_occur_window_size,
    ).build(bundle.products, warmup.warmup_samples)


def build_seed_graph_summary(project_root: Path, args: argparse.Namespace) -> dict:
    config, bundle = load_bundle_and_config(project_root, args)
    warmup = split_warmup(
        bundle.train_samples,
        mode=config.warmup_mode,
        warmup_ratio=config.warmup_ratio,
        warmup_count=config.warmup_count,
    )
    graph = SeedGraphBuilder(
        keyword_top_k=config.keyword_top_k,
        include_description=config.include_description,
        co_occur_window_size=config.co_occur_window_size,
    ).build(bundle.products, warmup.warmup_samples)
    semantic_memory = SemanticMemory.from_seed_graph(graph)

    test_only_titles = _collect_test_only_titles(bundle)
    inspected_title = args.inspect_title or (test_only_titles[0] if test_only_titles else warmup.warmup_samples[0].target)
    inspected_neighbors = _inspect_item_neighborhood(semantic_memory, inspected_title, limit=10)
    inspected_product = bundle.products.get(normalize_title(inspected_title))

    return {
        "dataset_name": bundle.dataset_name,
        "seed_graph": {
            "node_count": len(graph.nodes),
            "edge_count": len(graph.edges),
            "nodes_by_type": graph.node_count_by_type(),
            "edges_by_relation": graph.edge_count_by_relation(),
            "semantic_memory_edges": len(semantic_memory.edges),
            "semantic_memory_relations": semantic_memory.relation_counts(),
        },
        "warmup": {
            "samples": len(warmup.warmup_samples),
            "stream_samples": len(warmup.stream_samples),
            "mode": config.warmup_mode,
            "co_occur_window_size": config.co_occur_window_size,
            "keyword_top_k": config.keyword_top_k,
            "include_description": config.include_description,
        },
        "test_overlap": {
            "test_only_items_count": len(test_only_titles),
            "test_only_items_preview": test_only_titles[:10],
        },
        "inspection": {
            "title": inspected_title,
            "present_in_seed_graph": inspected_neighbors["item_present"],
            "metadata_exists": inspected_product is not None,
            "taxonomy_levels": inspected_product.taxonomy_levels if inspected_product else {},
            "keywords": (inspected_product.keywords[: config.keyword_top_k] if inspected_product else []),
            "outgoing_edges": inspected_neighbors["outgoing_edges"],
        },
    }


def build_phase3_train_summary(project_root: Path, args: argparse.Namespace) -> tuple[dict, LinearSemanticScorer]:
    config, bundle = load_bundle_and_config(project_root, args)
    warmup = split_warmup(
        bundle.train_samples,
        mode=config.warmup_mode,
        warmup_ratio=config.warmup_ratio,
        warmup_count=config.warmup_count,
    )
    graph = SeedGraphBuilder(
        keyword_top_k=config.keyword_top_k,
        include_description=config.include_description,
        co_occur_window_size=config.co_occur_window_size,
    ).build(bundle.products, warmup.warmup_samples)
    semantic_memory = SemanticMemory.from_seed_graph(graph)
    retriever = SemanticRetriever(semantic_memory)
    scorer = LinearSemanticScorer()

    # --- Dual optimization: neural + structural updates during training ---
    episodic_memory = EpisodicMemory(max_records=args.episodic_max_records)
    diagnostics = HybridDiagnostics(semantic_memory)
    writer = MemoryWriter(
        semantic_memory=semantic_memory,
        episodic_memory=episodic_memory,
        write_policy=WritePolicy(),
        config=MemoryWriterConfig(max_tentative_edges=args.max_tentative_edges),
    )

    train_result = train_semantic_scorer(
        samples=warmup.stream_samples,
        retriever=retriever,
        scorer=scorer,
        epochs=config.phase3_epochs,
        learning_rate=config.phase3_learning_rate,
        diagnostics=diagnostics,
        writer=writer,
        episodic_memory=episodic_memory,
    )
    summary = {
        "dataset_name": bundle.dataset_name,
        "phase": "phase3-train",
        "warmup_samples": len(warmup.warmup_samples),
        "train_stream_samples": len(warmup.stream_samples),
        "epochs": config.phase3_epochs,
        "learning_rate": config.phase3_learning_rate,
        "average_loss": train_result.average_loss,
        "metrics": train_result.metrics,
        "edit_counts": train_result.edit_counts,
        "weights": scorer.weights,
        "bias": scorer.bias,
    }
    return summary, scorer


def build_phase3_test_summary(project_root: Path, args: argparse.Namespace) -> dict:
    config, bundle = load_bundle_and_config(project_root, args)
    warmup = split_warmup(
        bundle.train_samples,
        mode=config.warmup_mode,
        warmup_ratio=config.warmup_ratio,
        warmup_count=config.warmup_count,
    )
    graph = SeedGraphBuilder(
        keyword_top_k=config.keyword_top_k,
        include_description=config.include_description,
        co_occur_window_size=config.co_occur_window_size,
    ).build(bundle.products, warmup.warmup_samples)
    semantic_memory = SemanticMemory.from_seed_graph(graph)
    retriever = SemanticRetriever(semantic_memory)

    if args.model_json:
        scorer = LinearSemanticScorer.load(str((project_root / args.model_json).resolve()))
    else:
        _, scorer = build_phase3_train_summary(project_root, args)

    test_result = evaluate_semantic_scorer(
        samples=bundle.test_samples,
        retriever=retriever,
        scorer=scorer,
        max_examples=10,
    )
    return {
        "dataset_name": bundle.dataset_name,
        "phase": "phase3-test",
        "test_samples": len(bundle.test_samples),
        "metrics": test_result.metrics,
        "examples": [
            {
                "sample_id": example.sample_id,
                "target": example.target,
                "top5": example.ranked_titles,
                "top_probability": example.top_probability,
            }
            for example in test_result.examples
        ],
        "weights": scorer.weights,
        "bias": scorer.bias,
    }


def build_phase4_test_summary(project_root: Path, args: argparse.Namespace) -> dict:
    config, bundle = load_bundle_and_config(project_root, args)
    warmup = split_warmup(
        bundle.train_samples,
        mode=config.warmup_mode,
        warmup_ratio=config.warmup_ratio,
        warmup_count=config.warmup_count,
    )
    graph = SeedGraphBuilder(
        keyword_top_k=config.keyword_top_k,
        include_description=config.include_description,
        co_occur_window_size=config.co_occur_window_size,
    ).build(bundle.products, warmup.warmup_samples)
    semantic_memory = SemanticMemory.from_seed_graph(graph)
    retriever = SemanticRetriever(semantic_memory)

    if args.model_json:
        scorer = LinearSemanticScorer.load(str((project_root / args.model_json).resolve()))
    else:
        _, scorer = build_phase3_train_summary(project_root, args)

    episodic_memory = EpisodicMemory(max_records=args.episodic_max_records)
    diagnostics = HybridDiagnostics(semantic_memory)
    writer = MemoryWriter(
        semantic_memory=semantic_memory,
        episodic_memory=episodic_memory,
        write_policy=WritePolicy(),
        config=MemoryWriterConfig(max_tentative_edges=args.max_tentative_edges),
    )
    test_result = evaluate_semantic_scorer_online(
        samples=bundle.test_samples,
        retriever=retriever,
        scorer=scorer,
        diagnostics=diagnostics,
        writer=writer,
        episodic_memory=episodic_memory,
        max_examples=10,
    )
    return {
        "dataset_name": bundle.dataset_name,
        "phase": "phase4-test-online",
        "test_samples": len(bundle.test_samples),
        "metrics": test_result.metrics,
        "edit_counts": test_result.edit_counts,
        "episodic_summary": test_result.episodic_summary,
        "semantic_summary": test_result.semantic_summary,
        "examples": [
            {
                "sample_id": example.sample_id,
                "target": example.target,
                "target_rank": example.target_rank,
                "outcome_type": example.outcome_type,
                "top5": example.ranked_titles,
                "top_probability": example.top_probability,
                "applied_edit_counts": example.applied_edit_counts,
            }
            for example in test_result.examples
        ],
        "audit_log_preview": test_result.audit_log[:10],
        "weights": scorer.weights,
        "bias": scorer.bias,
    }


def build_phase5_test_summary(project_root: Path, args: argparse.Namespace) -> dict:
    load_project_env(project_root)
    config, bundle = load_bundle_and_config(project_root, args)
    warmup = split_warmup(
        bundle.train_samples,
        mode=config.warmup_mode,
        warmup_ratio=config.warmup_ratio,
        warmup_count=config.warmup_count,
    )
    graph = SeedGraphBuilder(
        keyword_top_k=config.keyword_top_k,
        include_description=config.include_description,
        co_occur_window_size=config.co_occur_window_size,
    ).build(bundle.products, warmup.warmup_samples)
    semantic_memory = SemanticMemory.from_seed_graph(graph)
    retriever = SemanticRetriever(semantic_memory)

    if args.model_json:
        scorer = LinearSemanticScorer.load(str((project_root / args.model_json).resolve()))
    else:
        _, scorer = build_phase3_train_summary(project_root, args)

    samples = bundle.test_samples
    if args.max_test_samples > 0:
        samples = samples[: args.max_test_samples]

    episodic_memory = EpisodicMemory(max_records=args.episodic_max_records)
    diagnostics = HybridDiagnostics(semantic_memory)
    writer = MemoryWriter(
        semantic_memory=semantic_memory,
        episodic_memory=episodic_memory,
        write_policy=WritePolicy(),
        config=MemoryWriterConfig(max_tentative_edges=args.max_tentative_edges),
    )
    prompt_builder = Phase5PromptBuilder(project_root=project_root)
    model_manager = LanguageModelManager(provider=args.llm_provider)
    reranker = LLMReranker(
        provider=args.llm_provider,
        prompt_builder=prompt_builder,
        model_manager=model_manager,
    )
    lesson_agent = LLMLessonAgent(
        provider=args.llm_provider,
        prompt_builder=prompt_builder,
        model_manager=model_manager,
    )
    test_result = evaluate_llm_reranker_online(
        samples=samples,
        retriever=retriever,
        scorer=scorer,
        diagnostics=diagnostics,
        writer=writer,
        episodic_memory=episodic_memory,
        reranker=reranker,
        lesson_agent=lesson_agent,
        evidence_builder=LLMEvidenceBuilder(),
        max_examples=10,
    )
    return {
        "dataset_name": bundle.dataset_name,
        "phase": "phase5-test-llm",
        "llm_provider": args.llm_provider,
        "test_samples": len(samples),
        "full_test_samples": len(bundle.test_samples),
        "metrics": test_result.metrics,
        "parser_valid_rate": test_result.parser_valid_rate,
        "fallback_rate": test_result.fallback_rate,
        "lesson_valid_rate": test_result.lesson_valid_rate,
        "edit_counts": test_result.edit_counts,
        "episodic_summary": test_result.episodic_summary,
        "semantic_summary": test_result.semantic_summary,
        "examples": [
            {
                "sample_id": example.sample_id,
                "target": example.target,
                "target_rank": example.target_rank,
                "scorer_top5": example.scorer_top5,
                "llm_top5": example.llm_top5,
                "final_top5": example.final_top5,
                "fallback_used": example.fallback_used,
                "parser_valid": example.parser_valid,
                "lesson_valid": example.lesson_valid,
                "applied_edit_counts": example.applied_edit_counts,
            }
            for example in test_result.examples
        ],
        "audit_log_preview": test_result.audit_log[:10],
        "weights": scorer.weights,
        "bias": scorer.bias,
    }


def print_summary(summary: dict) -> None:
    print("=" * 72)
    print("CGM-Rec Phase 1 Dataset Inspection")
    print("=" * 72)
    print(f"Dataset               : {summary['dataset_name']}")
    print(f"Train samples         : {summary['counts']['train_samples']}")
    print(f"Test samples          : {summary['counts']['test_samples']}")
    print(f"Products              : {summary['counts']['products']}")
    print(f"Warm-up samples       : {summary['counts']['warmup_samples']}")
    print(f"Continual stream      : {summary['counts']['continual_stream_samples']}")
    print(f"Warm-up mode          : {summary['warmup']['mode']}")
    print(f"Chronological split   : {summary['warmup']['chronological']}")
    print("-" * 72)
    print(f"Example train sample  : {summary['examples']['train_sample_id']}")
    print(f"Target                : {summary['examples']['train_target']}")
    print(
        "Target index check    : "
        f"raw={summary['examples']['train_target_index']} "
        f"position={summary['examples']['train_target_position']} "
        f"base={summary['examples']['train_target_index_base']}"
    )
    print(f"Session preview       : {summary['examples']['train_session_items']}")
    print(f"Candidate preview     : {summary['examples']['train_candidate_items']}")
    print("-" * 72)
    print(f"Metadata example      : {summary['examples']['product_example']['title']}")
    print(f"Taxonomy              : {summary['examples']['product_example']['taxonomy_levels']}")
    print(f"Keywords              : {summary['examples']['product_example']['keywords']}")
    print(f"Description preview   : {summary['examples']['product_example']['description_preview']}")
    print("=" * 72)


def print_seed_graph_summary(summary: dict) -> None:
    print("=" * 72)
    print("CGM-Rec Phase 2 Seed Graph Inspection")
    print("=" * 72)
    print(f"Dataset               : {summary['dataset_name']}")
    print(f"Node count            : {summary['seed_graph']['node_count']}")
    print(f"Edge count            : {summary['seed_graph']['edge_count']}")
    print(f"Nodes by type         : {summary['seed_graph']['nodes_by_type']}")
    print(f"Edges by relation     : {summary['seed_graph']['edges_by_relation']}")
    print(f"Semantic memory edges : {summary['seed_graph']['semantic_memory_edges']}")
    print(f"Warm-up samples       : {summary['warmup']['samples']}")
    print(f"Continual stream      : {summary['warmup']['stream_samples']}")
    print(f"Keyword top-k         : {summary['warmup']['keyword_top_k']}")
    print(f"Use descriptions      : {summary['warmup']['include_description']}")
    print(f"Co-occur window       : {summary['warmup']['co_occur_window_size']}")
    print("-" * 72)
    print(f"Test-only item count  : {summary['test_overlap']['test_only_items_count']}")
    print(f"Preview new test items: {summary['test_overlap']['test_only_items_preview']}")
    print("-" * 72)
    print(f"Inspected title       : {summary['inspection']['title']}")
    print(f"Metadata exists       : {summary['inspection']['metadata_exists']}")
    print(f"Present in seed graph : {summary['inspection']['present_in_seed_graph']}")
    print(f"Taxonomy              : {summary['inspection']['taxonomy_levels']}")
    print(f"Keywords              : {summary['inspection']['keywords']}")
    print("Outgoing edges        :")
    for edge_text in summary["inspection"]["outgoing_edges"]:
        print(f"  - {edge_text}")
    print("=" * 72)


def print_phase3_train_summary(summary: dict) -> None:
    print("=" * 72)
    print("CGM-Rec Phase 3 Train Summary (Dual Optimization)")
    print("=" * 72)
    print(f"Dataset               : {summary['dataset_name']}")
    print(f"Warm-up samples       : {summary['warmup_samples']}")
    print(f"Train stream samples  : {summary['train_stream_samples']}")
    print(f"Epochs                : {summary['epochs']}")
    print(f"Learning rate         : {summary['learning_rate']}")
    print(f"Average loss          : {summary['average_loss']:.6f}")
    print(f"Metrics               : {summary['metrics']}")
    if summary.get("edit_counts"):
        print(f"Structural edits      : {summary['edit_counts']}")
    print("Weights               :")
    for feature_name, value in summary["weights"].items():
        print(f"  - {feature_name}: {value:.6f}")
    print(f"Bias                  : {summary['bias']:.6f}")
    print("=" * 72)


def print_phase3_test_summary(summary: dict) -> None:
    print("=" * 72)
    print("CGM-Rec Phase 3 Test Summary")
    print("=" * 72)
    print(f"Dataset               : {summary['dataset_name']}")
    print(f"Test samples          : {summary['test_samples']}")
    print(f"Metrics               : {summary['metrics']}")
    print("Example predictions   :")
    for example in summary["examples"]:
        print(
            f"  - {example['sample_id']}: target={example['target']} "
            f"top5={example['top5']} p={example['top_probability']:.6f}"
        )
    print("=" * 72)


def print_phase4_test_summary(summary: dict) -> None:
    print("=" * 72)
    print("CGM-Rec Phase 4 Online Test Summary")
    print("=" * 72)
    print(f"Dataset               : {summary['dataset_name']}")
    print(f"Test samples          : {summary['test_samples']}")
    print(f"Metrics               : {summary['metrics']}")
    print(f"Edit counts           : {summary['edit_counts']}")
    print(f"Episodic summary      : {summary['episodic_summary']}")
    print(f"Semantic summary      : {summary['semantic_summary']}")
    print("Example sessions      :")
    for example in summary["examples"]:
        print(
            f"  - {example['sample_id']}: target={example['target']} rank={example['target_rank']} "
            f"outcome={example['outcome_type']} top5={example['top5']} "
            f"edits={example['applied_edit_counts']}"
        )
    print("=" * 72)


def print_phase5_test_summary(summary: dict) -> None:
    print("=" * 72)
    print("CGM-Rec Phase 5 LLM Online Test Summary")
    print("=" * 72)
    print(f"Dataset               : {summary['dataset_name']}")
    print(f"LLM provider          : {summary['llm_provider']}")
    print(f"Test samples          : {summary['test_samples']} / {summary['full_test_samples']}")
    print(f"Metrics               : {summary['metrics']}")
    print(f"Parser valid rate     : {summary['parser_valid_rate']:.4f}")
    print(f"Fallback rate         : {summary['fallback_rate']:.4f}")
    print(f"Lesson valid rate     : {summary['lesson_valid_rate']:.4f}")
    print(f"Edit counts           : {summary['edit_counts']}")
    print(f"Episodic summary      : {summary['episodic_summary']}")
    print(f"Semantic summary      : {summary['semantic_summary']}")
    print("Example sessions      :")
    for example in summary["examples"]:
        print(
            f"  - {example['sample_id']}: target={example['target']} rank={example['target_rank']} "
            f"fallback={example['fallback_used']} parser={example['parser_valid']} "
            f"final_top5={example['final_top5']}"
        )
    print("=" * 72)


def _collect_test_only_titles(bundle) -> list[str]:
    train_titles = set()
    test_titles = set()
    for sample in bundle.train_samples:
        train_titles.update(sample.parsed_input.session_items)
        train_titles.update(sample.parsed_input.candidate_items)
        train_titles.add(sample.target)
    for sample in bundle.test_samples:
        test_titles.update(sample.parsed_input.session_items)
        test_titles.update(sample.parsed_input.candidate_items)
        test_titles.add(sample.target)
    return sorted(test_titles - train_titles)


def _inspect_item_neighborhood(memory: SemanticMemory, title: str, limit: int = 10) -> dict:
    item_id = f"item::{normalize_title(title)}"
    item_present = item_id in memory.nodes
    outgoing = []
    for edge in memory.get_edges(item_id)[:limit]:
        dst_label = memory.nodes.get(edge.dst).label if edge.dst in memory.nodes else edge.dst
        outgoing.append(
            f"{edge.relation} -> {dst_label} "
            f"(weight={edge.weight:.2f}, conf={edge.confidence:.2f}, support={edge.support_count}, source={edge.source_kind})"
        )
    return {"item_present": item_present, "outgoing_edges": outgoing}


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent

    scorer_to_save: LinearSemanticScorer | None = None

    if args.view == "data":
        summary = build_summary(project_root, args)
        print_summary(summary)
    elif args.view == "seed-graph":
        summary = build_seed_graph_summary(project_root, args)
        print_seed_graph_summary(summary)
        if args.dump_graph_json or args.dump_graph_text:
            graph = build_seed_graph(project_root, args)
            if args.dump_graph_json:
                output_path = project_root / args.dump_graph_json
                write_json(output_path, graph.to_dict())
                print(f"Saved full seed graph JSON to {output_path}")
            if args.dump_graph_text:
                output_path = project_root / args.dump_graph_text
                write_text(output_path, graph.to_text())
                print(f"Saved full seed graph text to {output_path}")
    elif args.view == "phase3-train":
        summary, scorer_to_save = build_phase3_train_summary(project_root, args)
        print_phase3_train_summary(summary)
    elif args.view == "phase3-test":
        summary = build_phase3_test_summary(project_root, args)
        print_phase3_test_summary(summary)
    elif args.view == "phase4-test-online":
        summary = build_phase4_test_summary(project_root, args)
        print_phase4_test_summary(summary)
    else:
        summary = build_phase5_test_summary(project_root, args)
        print_phase5_test_summary(summary)

    if args.model_json and scorer_to_save is not None:
        model_path = project_root / args.model_json
        scorer_to_save.save(str(model_path))
        print(f"Saved scorer weights to {model_path}")

    if args.output_json:
        output_path = project_root / args.output_json
        write_json(output_path, summary)
        print(f"Saved summary JSON to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
