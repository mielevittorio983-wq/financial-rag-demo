from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from reg_rag.manifest import audit_manifest
from reg_rag.official_mcq import import_official_mcq, run_official_mcq_eval
from reg_rag.official_robustness import build_official_robustness_set
from reg_rag.answer import answer_qa_file, eval_answers
from reg_rag.claim_eval import eval_claims
from reg_rag.eval import eval_retrieval, write_eval_report
from reg_rag.eval_diagnostics import diagnose_cfqa_retrieval, diagnose_formal_report
from reg_rag.external import download_cfqa_reports, filter_cfqa_ready_qa, import_cfqa
from reg_rag.io import write_jsonl
from reg_rag.io import read_jsonl
from reg_rag.llm_prompt import build_grounded_prompt
from reg_rag.parsers import parse_file
from reg_rag.parsers.legacy_office import preconvert_legacy_office
from reg_rag.pipeline import parse_manifest
from reg_rag.reports import compare_eval_reports, summarize_answer_failures
from reg_rag.retrieval import retrieve_records
from reg_rag.synthetic import build_synthetic_dataset
from reg_rag.table_summary import add_table_summary_records
from reg_rag.formal_eval import run_ablation_suite, run_formal_eval, write_formal_outputs
from reg_rag.agent_pipeline import run_agent
from reg_rag.freeze import freeze_official_baseline
from reg_rag.unseen import build_unseen_document_manifest
from reg_rag.unseen_benchmark import (
    export_unseen_review_sheet,
    freeze_unseen_benchmark,
    import_unseen_review_sheet,
    package_unseen_benchmark,
)
from reg_rag.unseen_mcq import (
    assemble_balanced_candidate_slate,
    assemble_challenge_candidate_slate,
    assemble_full_challenge_slate,
    derive_numeric_mcq_candidates,
)
from reg_rag.unseen_reasoning import derive_ambiguity_refusal_candidates, derive_cross_file_calculation_candidates
from reg_rag.unseen_version import derive_version_conflict_candidates
from reg_rag.unseen_review_workflow import merge_unseen_review_packets, prepare_unseen_review_packets
from reg_rag.unseen_source_audit import audit_unseen_candidates
from reg_rag.unseen_seed import generate_unseen_fact_seeds
from reg_rag.unseen_sealed_eval import run_unseen_sealed, score_unseen_sealed


GENERATOR_CHOICES = ["rule", "mock", "mock_llm", "openai"]


def _default_generator() -> str:
    value = os.getenv("REG_RAG_LLM_ADAPTER", "rule")
    return value if value in GENERATOR_CHOICES else "rule"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reg-rag")
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit-manifest", help="Validate manifest rows and local files.")
    audit.add_argument("manifest", type=Path)
    audit.add_argument("--base-dir", type=Path, default=None)
    audit.add_argument("--out", type=Path, default=None)

    parse = sub.add_parser("parse-file", help="Parse one DOCX/PDF/XLSX file into normalized JSONL.")
    parse.add_argument("path", type=Path)
    parse.add_argument("--doc-id", default=None)
    parse.add_argument("--title", default=None)
    parse.add_argument("--out", type=Path, required=True)

    parse_all = sub.add_parser("parse-manifest", help="Parse all supported files from a manifest into one JSONL.")
    parse_all.add_argument("manifest", type=Path)
    parse_all.add_argument("--base-dir", type=Path, default=None)
    parse_all.add_argument("--out", type=Path, required=True)

    legacy = sub.add_parser("preconvert-legacy", help="Batch-convert DOC/XLS files into the hash cache.")
    legacy.add_argument("root", type=Path)
    legacy.add_argument("--batch-size", type=int, default=32)
    legacy.add_argument("--out", type=Path, default=None)

    freeze = sub.add_parser("freeze-official", help="Freeze official QA, corpus, code and config hashes.")
    freeze.add_argument("questions", type=Path)
    freeze.add_argument("gold", type=Path)
    freeze.add_argument("dataset_root", type=Path)
    freeze.add_argument("--out", type=Path, required=True)
    freeze.add_argument("--semantic-index", type=Path, default=None)
    freeze.add_argument("--freeze-name", default="official_300_full500_freeze_v2")

    index = sub.add_parser("build-official-index", help="Build a persistent SQLite semantic feature index.")
    index.add_argument("catalog_root", type=Path)
    index.add_argument("--out", type=Path, required=True)

    unseen = sub.add_parser("make-unseen-manifest", help="Select frozen documents never used by official QA.")
    unseen.add_argument("questions", type=Path)
    unseen.add_argument("catalog_root", type=Path)
    unseen.add_argument("--limit", type=int, default=100)
    unseen.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    unseen.add_argument("--out", type=Path, required=True)

    unseen_pack = sub.add_parser("package-unseen-benchmark", help="Split reviewed unseen drafts into release and private gold.")
    unseen_pack.add_argument("drafts", type=Path)
    unseen_pack.add_argument("--out-dir", type=Path, required=True)

    unseen_seed = sub.add_parser("generate-unseen-seeds", help="Generate private review-only fact seeds from unseen documents.")
    unseen_seed.add_argument("manifest", type=Path)
    unseen_seed.add_argument("--per-document", type=int, default=2)
    unseen_seed.add_argument("--target-count", type=int, default=None)
    unseen_seed.add_argument("--out", type=Path, required=True)

    unseen_review = sub.add_parser("export-unseen-review", help="Export private unseen drafts for human review.")
    unseen_review.add_argument("drafts", type=Path)
    unseen_review.add_argument("--out", type=Path, required=True)

    unseen_review_import = sub.add_parser("import-unseen-review", help="Merge a reviewed private CSV back into JSONL.")
    unseen_review_import.add_argument("drafts", type=Path)
    unseen_review_import.add_argument("review", type=Path)
    unseen_review_import.add_argument("--out", type=Path, required=True)

    unseen_mcq = sub.add_parser("derive-unseen-mcq", help="Create review-only numeric MCQ candidates.")
    unseen_mcq.add_argument("drafts", type=Path)
    unseen_mcq.add_argument("--limit", type=int, default=100)
    unseen_mcq.add_argument("--out", type=Path, required=True)

    unseen_slate = sub.add_parser("assemble-unseen-slate", help="Build a 100-open/100-MCQ non-overlapping slate.")
    unseen_slate.add_argument("open_drafts", type=Path)
    unseen_slate.add_argument("mcq_drafts", type=Path)
    unseen_slate.add_argument("--out", type=Path, required=True)

    unseen_cross = sub.add_parser("derive-unseen-cross-file", help="Create review-only two-source calculation candidates.")
    unseen_cross.add_argument("drafts", type=Path)
    unseen_cross.add_argument("--limit", type=int, default=20)
    unseen_cross.add_argument("--out", type=Path, required=True)

    unseen_challenge = sub.add_parser(
        "assemble-unseen-challenge-slate",
        help="Build a non-overlapping slate with open, MCQ and cross-file questions.",
    )
    unseen_challenge.add_argument("base_drafts", type=Path)
    unseen_challenge.add_argument("mcq_drafts", type=Path)
    unseen_challenge.add_argument("cross_file_drafts", type=Path)
    unseen_challenge.add_argument("--out", type=Path, required=True)

    unseen_refusal = sub.add_parser("derive-unseen-refusals", help="Create review-only ambiguity refusal candidates.")
    unseen_refusal.add_argument("drafts", type=Path)
    unseen_refusal.add_argument("--limit", type=int, default=20)
    unseen_refusal.add_argument("--out", type=Path, required=True)

    unseen_version = sub.add_parser("derive-unseen-versions", help="Create review-only version conflict candidates.")
    unseen_version.add_argument("manifest", type=Path)
    unseen_version.add_argument("--limit", type=int, default=10)
    unseen_version.add_argument("--out", type=Path, required=True)

    unseen_full_slate = sub.add_parser("assemble-unseen-full-slate", help="Build the final quota-complete private slate.")
    unseen_full_slate.add_argument("base_drafts", type=Path)
    unseen_full_slate.add_argument("mcq_drafts", type=Path)
    unseen_full_slate.add_argument("cross_file_drafts", type=Path)
    unseen_full_slate.add_argument("refusal_drafts", type=Path)
    unseen_full_slate.add_argument("version_drafts", type=Path)
    unseen_full_slate.add_argument("--exclude-audit", type=Path, action="append", default=[])
    unseen_full_slate.add_argument("--out", type=Path, required=True)

    unseen_packets = sub.add_parser("prepare-unseen-review-packets", help="Split private drafts into two review packets.")
    unseen_packets.add_argument("drafts", type=Path)
    unseen_packets.add_argument("--out-dir", type=Path, required=True)

    unseen_merge = sub.add_parser("merge-unseen-review-packets", help="Merge two independent review packets by consensus.")
    unseen_merge.add_argument("drafts", type=Path)
    unseen_merge.add_argument("reviewer_a", type=Path)
    unseen_merge.add_argument("reviewer_b", type=Path)
    unseen_merge.add_argument("--out", type=Path, required=True)

    unseen_source_audit = sub.add_parser("audit-unseen-sources", help="Verify every private candidate against source files.")
    unseen_source_audit.add_argument("drafts", type=Path)
    unseen_source_audit.add_argument("--reviewer", default="codex_authorized_source_audit")
    unseen_source_audit.add_argument("--out-dir", type=Path, required=True)

    unseen_freeze = sub.add_parser("freeze-unseen-benchmark", help="Freeze reviewed release/gold hashes and audit identity.")
    unseen_freeze.add_argument("questions", type=Path)
    unseen_freeze.add_argument("gold", type=Path)
    unseen_freeze.add_argument("source_manifest", type=Path)
    unseen_freeze.add_argument("source_audit_report", type=Path)
    unseen_freeze.add_argument("--freeze-name", default="unseen_100_codex_review_v1")
    unseen_freeze.add_argument("--semantic-index", type=Path, default=None)
    unseen_freeze.add_argument("--out", type=Path, required=True)

    unseen_run = sub.add_parser("run-unseen-sealed", help="Run frozen unseen release without loading private gold.")
    unseen_run.add_argument("questions", type=Path)
    unseen_run.add_argument("semantic_index", type=Path)
    unseen_run.add_argument("--out-dir", type=Path, required=True)
    unseen_run.add_argument("--top-sources", type=int, default=5)
    unseen_run.add_argument("--generator", choices=GENERATOR_CHOICES, default=_default_generator())

    unseen_score = sub.add_parser("score-unseen-sealed", help="Score already materialized sealed predictions.")
    unseen_score.add_argument("predictions", type=Path)
    unseen_score.add_argument("gold", type=Path)
    unseen_score.add_argument("--out-dir", type=Path, required=True)

    table_summary = sub.add_parser("add-table-summaries", help="Append generated table summary records to parsed JSONL.")
    table_summary.add_argument("records", type=Path)
    table_summary.add_argument("--out", type=Path, required=True)
    table_summary.add_argument("--max-rows-per-summary", type=int, default=32)

    search = sub.add_parser("search-jsonl", help="Run a lightweight lexical search over parsed JSONL records.")
    search.add_argument("records", type=Path)
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--mode", choices=["simple", "bm25", "vector", "hybrid"], default="simple")
    search.add_argument("--reranker", choices=["none", "heuristic", "page_aware", "section_aware", "table_aware"], default="none")

    synth = sub.add_parser("make-synthetic", help="Create a small repeatable synthetic dataset.")
    synth.add_argument("--out-dir", type=Path, default=Path("data/synthetic"))

    cfqa = sub.add_parser("import-cfqa", help="Convert downloaded CFQA split files into reg-rag QA and manifest files.")
    cfqa.add_argument("--source-dir", type=Path, default=Path("data/external/cfqa/source"))
    cfqa.add_argument("--out-dir", type=Path, default=Path("data/external/cfqa/split_by_company_test"))
    cfqa.add_argument("--split", choices=["split_by_company", "split_by_year"], default="split_by_company")
    cfqa.add_argument("--subset", choices=["train", "dev", "test"], default="test")
    cfqa.add_argument("--limit", type=int, default=None)
    cfqa.add_argument("--offset", type=int, default=0)

    official_import = sub.add_parser("import-official-mcq", help="Split the official MCQ workbook into runtime questions and isolated gold files.")
    official_import.add_argument("qa", type=Path)
    official_import.add_argument("dataset_root", type=Path)
    official_import.add_argument("--out-dir", type=Path, required=True)

    official_eval = sub.add_parser("eval-official-mcq", help="Run the source-routed official multiple-choice benchmark.")
    official_eval.add_argument("questions", type=Path)
    official_eval.add_argument("gold", type=Path)
    official_eval.add_argument("--out-dir", type=Path, required=True)
    official_eval.add_argument("--routing", choices=["direct", "corpus", "semantic"], default="direct")
    official_eval.add_argument("--catalog-root", type=Path, default=None)
    official_eval.add_argument("--semantic-index", type=Path, default=None)

    official_robustness = sub.add_parser(
        "make-official-robustness",
        help="Create deterministic paraphrased official questions without reading gold.",
    )
    official_robustness.add_argument("questions", type=Path)
    official_robustness.add_argument("--out", type=Path, required=True)
    official_robustness.add_argument(
        "--variant",
        choices=["template_paraphrase_v1", "title_redacted_v1", "semantic_query_v1"],
        default="template_paraphrase_v1",
    )

    dl_cfqa = sub.add_parser("download-cfqa-reports", help="Download CFQA annual report PDFs from public CNINFO announcements.")
    dl_cfqa.add_argument("manifest", type=Path)
    dl_cfqa.add_argument("--limit", type=int, default=None)
    dl_cfqa.add_argument("--offset", type=int, default=0)
    dl_cfqa.add_argument("--sleep", type=float, default=0.5)

    filter_cfqa = sub.add_parser("filter-cfqa-ready-qa", help="Create a QA JSONL containing only CFQA questions whose PDFs are ready.")
    filter_cfqa.add_argument("qa", type=Path)
    filter_cfqa.add_argument("manifest", type=Path)
    filter_cfqa.add_argument("--out", type=Path, required=True)

    eval_cmd = sub.add_parser("eval-retrieval", help="Evaluate retrieval on a QA JSONL seed set.")
    eval_cmd.add_argument("records", type=Path)
    eval_cmd.add_argument("qa", type=Path)
    eval_cmd.add_argument("--top-k", type=int, default=3)
    eval_cmd.add_argument("--mode", choices=["simple", "bm25", "vector", "hybrid"], default="simple")
    eval_cmd.add_argument("--reranker", choices=["none", "heuristic", "page_aware", "section_aware", "table_aware"], default="none")
    eval_cmd.add_argument("--out", type=Path, default=None)

    answer_cmd = sub.add_parser("answer-qa", help="Generate rule-based baseline answers for a QA JSONL file.")
    answer_cmd.add_argument("records", type=Path)
    answer_cmd.add_argument("qa", type=Path)
    answer_cmd.add_argument("--top-k", type=int, default=8)
    answer_cmd.add_argument("--mode", choices=["simple", "bm25", "vector", "hybrid"], default="hybrid")
    answer_cmd.add_argument("--generator", choices=GENERATOR_CHOICES, default=_default_generator())
    answer_cmd.add_argument("--reranker", choices=["none", "heuristic", "page_aware", "section_aware", "table_aware"], default="none")
    answer_cmd.add_argument("--out", type=Path, required=True)

    prompt_cmd = sub.add_parser("build-prompt", help="Build a grounded LLM prompt from retrieved evidence.")
    prompt_cmd.add_argument("records", type=Path)
    prompt_cmd.add_argument("question")
    prompt_cmd.add_argument("--top-k", type=int, default=8)
    prompt_cmd.add_argument("--mode", choices=["simple", "bm25", "vector", "hybrid"], default="hybrid")
    prompt_cmd.add_argument("--reranker", choices=["none", "heuristic", "page_aware", "section_aware", "table_aware"], default="none")
    prompt_cmd.add_argument("--out", type=Path, default=None)

    eval_answer_cmd = sub.add_parser("eval-answer", help="Evaluate generated answer JSONL against QA JSONL.")
    eval_answer_cmd.add_argument("predictions", type=Path)
    eval_answer_cmd.add_argument("qa", type=Path)
    eval_answer_cmd.add_argument("--out", type=Path, default=None)

    eval_claim_cmd = sub.add_parser("eval-claims", help="Evaluate whether answer claims are supported by cited evidence.")
    eval_claim_cmd.add_argument("predictions", type=Path)
    eval_claim_cmd.add_argument("--out", type=Path, default=None)

    compare_cmd = sub.add_parser("compare-evals", help="Render a Markdown comparison for retrieval eval JSON reports.")
    compare_cmd.add_argument("reports", type=Path, nargs="+")
    compare_cmd.add_argument("--out", type=Path, required=True)

    summarize_failures_cmd = sub.add_parser("summarize-answer-failures", help="Summarize generated answer failure categories.")
    summarize_failures_cmd.add_argument("answer_report", type=Path)
    summarize_failures_cmd.add_argument("--claim-report", type=Path, default=None)
    summarize_failures_cmd.add_argument("--out", type=Path, required=True)

    formal_cmd = sub.add_parser("formal-eval", help="Run blind/adversarial retrieval, answer, citation and refusal evaluation.")
    formal_cmd.add_argument("records", type=Path)
    formal_cmd.add_argument("qa", type=Path)
    formal_cmd.add_argument("--mode", choices=["simple", "bm25", "vector", "hybrid"], default="hybrid")
    formal_cmd.add_argument("--reranker", choices=["none", "heuristic", "page_aware", "section_aware", "table_aware"], default="table_aware")
    formal_cmd.add_argument("--top-k", type=int, default=5)
    formal_cmd.add_argument("--generator", choices=GENERATOR_CHOICES, default=_default_generator())
    formal_cmd.add_argument("--check-hallucination", action="store_true")
    formal_cmd.add_argument("--out-dir", type=Path, required=True)

    ablation_cmd = sub.add_parser("run-ablations", help="Run the formal retrieval and trust ablation suite.")
    ablation_cmd.add_argument("records", type=Path)
    ablation_cmd.add_argument("qa", type=Path)
    ablation_cmd.add_argument("--out-dir", type=Path, required=True)

    diagnose_cmd = sub.add_parser("diagnose-formal-eval", help="Analyze gold-rank, refusal and calculation bottlenecks from a formal eval report.")
    diagnose_cmd.add_argument("report", type=Path)
    diagnose_cmd.add_argument("--out-dir", type=Path, required=True)

    cfqa_diag_cmd = sub.add_parser("diagnose-cfqa-retrieval", help="Diagnose a CFQA retrieval report along template x year dimensions.")
    cfqa_diag_cmd.add_argument("report", type=Path)
    cfqa_diag_cmd.add_argument("qa", type=Path)
    cfqa_diag_cmd.add_argument("--out-dir", type=Path, required=True)

    serve_cmd = sub.add_parser("serve", help="Run the FastAPI query service.")
    serve_cmd.add_argument("--records", type=Path, default=Path("data/synthetic/records.jsonl"))
    serve_cmd.add_argument("--host", default=os.getenv("REG_RAG_HOST", "0.0.0.0"))
    serve_cmd.add_argument("--port", type=int, default=int(os.getenv("PORT", "7860")))

    agent_cmd = sub.add_parser("agent", help="Run Agent self-check pipeline for a single question, outputting full step traces.")
    agent_cmd.add_argument("records", type=Path, help="records.jsonl path")
    agent_cmd.add_argument("question", help="Question text (quote if contains spaces)")
    agent_cmd.add_argument("--top-k", type=int, default=8)
    agent_cmd.add_argument("--generator", choices=GENERATOR_CHOICES, default=_default_generator())
    agent_cmd.add_argument("--out", type=Path, default=None, help="Output JSON path")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "audit-manifest":
        report = audit_manifest(args.manifest, base_dir=args.base_dir)
        text = json.dumps(report, ensure_ascii=False, indent=2)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text, encoding="utf-8")
        print(text)
        return 0

    if args.command == "parse-file":
        records = parse_file(args.path, doc_id=args.doc_id, title=args.title)
        write_jsonl(args.out, records)
        print(json.dumps({"out": str(args.out), "records": len(records)}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "parse-manifest":
        report = parse_manifest(args.manifest, out_path=args.out, base_dir=args.base_dir)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "add-table-summaries":
        records = read_jsonl(args.records)
        enriched = add_table_summary_records(records, max_rows_per_summary=args.max_rows_per_summary)
        write_jsonl(args.out, enriched)
        summary_records = len(enriched) - len(records)
        print(
            json.dumps(
                {
                    "out": str(args.out),
                    "input_records": len(records),
                    "output_records": len(enriched),
                    "summary_records": summary_records,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "search-jsonl":
        records = read_jsonl(args.records)
        hits = retrieve_records(records, args.query, top_k=args.top_k, mode=args.mode, reranker=args.reranker)
        payload = [
            {
                "score": hit.score,
                "locator": hit.record.get("chunk_id") or hit.record.get("cell_id"),
                "text": hit.record.get("text") or hit.record.get("value"),
                "local_path": hit.record.get("local_path"),
            }
            for hit in hits
        ]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "make-synthetic":
        report = build_synthetic_dataset(args.out_dir)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "import-cfqa":
        report = import_cfqa(args.source_dir, args.out_dir, split=args.split, subset=args.subset, limit=args.limit, offset=args.offset)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "preconvert-legacy":
        paths = [
            path
            for path in args.root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".doc", ".xls"} and "__MACOSX" not in path.parts
        ]
        report = preconvert_legacy_office(paths, batch_size=args.batch_size)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "freeze-official":
        report = freeze_official_baseline(
            args.questions,
            args.gold,
            args.dataset_root,
            args.out,
            semantic_index=args.semantic_index,
            freeze_name=args.freeze_name,
        )
        print(json.dumps({key: report[key] for key in ("dataset_file_count", "git_head", "git_dirty")}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "build-official-index":
        from reg_rag.official_index import build_official_semantic_index

        report = build_official_semantic_index(args.catalog_root, args.out)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "make-unseen-manifest":
        report = build_unseen_document_manifest(
            args.questions,
            args.catalog_root,
            args.out,
            limit=args.limit,
            exclude_manifests=args.exclude_manifest,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "package-unseen-benchmark":
        report = package_unseen_benchmark(args.drafts, args.out_dir)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "generate-unseen-seeds":
        report = generate_unseen_fact_seeds(
            args.manifest,
            args.out,
            per_document=args.per_document,
            target_count=args.target_count,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "export-unseen-review":
        report = export_unseen_review_sheet(args.drafts, args.out)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "import-unseen-review":
        report = import_unseen_review_sheet(args.drafts, args.review, args.out)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "derive-unseen-mcq":
        report = derive_numeric_mcq_candidates(args.drafts, args.out, limit=args.limit)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "assemble-unseen-slate":
        report = assemble_balanced_candidate_slate(args.open_drafts, args.mcq_drafts, args.out)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "derive-unseen-cross-file":
        report = derive_cross_file_calculation_candidates(args.drafts, args.out, limit=args.limit)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "assemble-unseen-challenge-slate":
        report = assemble_challenge_candidate_slate(
            args.base_drafts,
            args.mcq_drafts,
            args.cross_file_drafts,
            args.out,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "derive-unseen-refusals":
        report = derive_ambiguity_refusal_candidates(args.drafts, args.out, limit=args.limit)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "derive-unseen-versions":
        report = derive_version_conflict_candidates(args.manifest, args.out, limit=args.limit)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "assemble-unseen-full-slate":
        excluded_ids = set()
        for audit_path in args.exclude_audit:
            excluded_ids.update(
                str(item.get("id") or "")
                for item in read_jsonl(audit_path)
                if item.get("decision") != "accepted"
            )
        report = assemble_full_challenge_slate(
            args.base_drafts,
            args.mcq_drafts,
            args.cross_file_drafts,
            args.refusal_drafts,
            args.version_drafts,
            args.out,
            excluded_ids=excluded_ids,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "prepare-unseen-review-packets":
        report = prepare_unseen_review_packets(args.drafts, args.out_dir)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "merge-unseen-review-packets":
        report = merge_unseen_review_packets(args.drafts, args.reviewer_a, args.reviewer_b, args.out)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "audit-unseen-sources":
        report = audit_unseen_candidates(args.drafts, args.out_dir, reviewer=args.reviewer)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "freeze-unseen-benchmark":
        report = freeze_unseen_benchmark(
            args.questions,
            args.gold,
            args.source_manifest,
            args.source_audit_report,
            args.out,
            freeze_name=args.freeze_name,
            semantic_index=args.semantic_index,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "run-unseen-sealed":
        report = run_unseen_sealed(
            args.questions,
            args.semantic_index,
            args.out_dir,
            top_sources=args.top_sources,
            generator=args.generator,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "score-unseen-sealed":
        report = score_unseen_sealed(args.predictions, args.gold, args.out_dir)
        summary = {key: value for key, value in report.items() if key != "rows"}
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "import-official-mcq":
        report = import_official_mcq(args.qa, args.dataset_root, args.out_dir)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "eval-official-mcq":
        report = run_official_mcq_eval(
            args.questions,
            args.gold,
            args.out_dir,
            routing=args.routing,
            catalog_root=args.catalog_root,
            semantic_index=args.semantic_index,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "make-official-robustness":
        report = build_official_robustness_set(args.questions, args.out, variant=args.variant)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "download-cfqa-reports":
        report = download_cfqa_reports(args.manifest, limit=args.limit, offset=args.offset, sleep_seconds=args.sleep)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "filter-cfqa-ready-qa":
        report = filter_cfqa_ready_qa(args.qa, args.manifest, args.out)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "eval-retrieval":
        report = eval_retrieval(args.records, args.qa, top_k=args.top_k, mode=args.mode, reranker=args.reranker)
        if args.out:
            write_eval_report(report, args.out)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "answer-qa":
        report = answer_qa_file(args.records, args.qa, args.out, top_k=args.top_k, mode=args.mode, generator=args.generator, reranker=args.reranker)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "build-prompt":
        records = read_jsonl(args.records)
        hits = retrieve_records(records, args.question, top_k=args.top_k, mode=args.mode, reranker=args.reranker)
        evidence = [
            {
                "locator": hit.record.get("chunk_id") or hit.record.get("cell_id"),
                "doc_id": hit.record.get("doc_id"),
                "text": hit.record.get("text") or hit.record.get("value"),
                "score": hit.score,
            }
            for hit in hits[: min(5, len(hits))]
        ]
        prompt = build_grounded_prompt(args.question, evidence)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(prompt, encoding="utf-8")
        print(prompt)
        return 0

    if args.command == "eval-answer":
        report = eval_answers(args.predictions, args.qa)
        text = json.dumps(report, ensure_ascii=False, indent=2)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text, encoding="utf-8")
        print(text)
        return 0

    if args.command == "eval-claims":
        report = eval_claims(args.predictions)
        text = json.dumps(report, ensure_ascii=False, indent=2)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text, encoding="utf-8")
        print(text)
        return 0

    if args.command == "compare-evals":
        report = compare_eval_reports(args.reports, args.out)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "summarize-answer-failures":
        report = summarize_answer_failures(args.answer_report, args.out, claim_report_path=args.claim_report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "formal-eval":
        report = run_formal_eval(
            args.records,
            args.qa,
            mode=args.mode,
            reranker=args.reranker,
            top_k=args.top_k,
            generator=args.generator,
            check_hallucination=args.check_hallucination,
        )
        write_formal_outputs(report, args.out_dir)
        print(json.dumps({key: value for key, value in report.items() if key != "rows" and key != "by_type"}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "run-ablations":
        report = run_ablation_suite(args.records, args.qa, args.out_dir)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "diagnose-formal-eval":
        report = diagnose_formal_report(args.report, args.out_dir)
        print(json.dumps({key: value for key, value in report.items() if key not in {"gold_position_by_split"}}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "diagnose-cfqa-retrieval":
        report = diagnose_cfqa_retrieval(args.report, args.qa, args.out_dir)
        keys = {"matrix"}
        print(json.dumps({k: v for k, v in report.items() if k not in keys}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "serve":
        import uvicorn

        from reg_rag.api import create_app

        uvicorn.run(create_app(args.records), host=args.host, port=args.port)
        return 0

    if args.command == "agent":
        records = read_jsonl(args.records)
        result = run_agent(records, args.question, top_k=args.top_k, generator=args.generator)
        output = result.to_dict()
        text = json.dumps(output, ensure_ascii=False, indent=2)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text, encoding="utf-8")
        print(text)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
