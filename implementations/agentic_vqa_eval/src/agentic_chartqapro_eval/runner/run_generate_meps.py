r"""Runner: generate Model Evaluation Packets (MEPs) for ChartQAPro.

Usage:
    uv run --env-file .env -m agentic_chartqapro_eval.runner.run_generate_meps \\
        --dataset chartqapro \\
        --split test \\
        --n 200 \\
        --config openai_gemini \\
        --workers 8 \\
        --out meps/
"""

import argparse
import contextlib
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from ..agents.planner_agent import PlannerAgent
from ..agents.verifier_agent import VerifierAgent
from ..agents.vision_agent import VisionAgent
from ..datasets.chartqapro_loader import load_chartqapro
from ..datasets.perceived_sample import PerceivedSample
from ..langfuse_integration.client import get_client
from ..langfuse_integration.dataset import register_dataset
from ..langfuse_integration.prompts import push_prompts
from ..langfuse_integration.tracing import (
    log_trace_scores,
    sample_trace,
)
from ..mep.schema import (
    MEP,
    ImageRef,
    MEPConfig,
    MEPOcr,
    MEPPlan,
    MEPSample,
    MEPTimestamps,
    MEPVerifier,
    MEPVision,
)
from ..mep.writer import write_mep
from ..tools.ocr_reader_tool import OcrReaderTool
from ..utils.hashing import sha256_file
from ..utils.json_strict import parse_strict
from ..utils.timing import iso_now, timed


load_dotenv()

# ---------------------------------------------------------------------------
# Backend configuration presets
# ---------------------------------------------------------------------------

BACKEND_CONFIGS: dict = {
    "openai_openai": {
        "planner_backend": "openai",
        "planner_model": "gpt-4o",
        "vision_backend": "openai",
        "vision_model": "gpt-4o",
        "judge_backend": "openai",
    },
    "gemini_gemini": {
        "planner_backend": "gemini",
        "planner_model": "gemini-2.5-flash-lite",
        "vision_backend": "gemini",
        "vision_model": "gemini-2.5-flash-lite",
        "judge_backend": "gemini",
    },
    "openai_gemini": {
        "planner_backend": "openai",
        "planner_model": "gpt-4o",
        "vision_backend": "gemini",
        "vision_model": "gemini-2.5-flash-lite",
        "judge_backend": "openai",
    },
    "gemini_openai": {
        "planner_backend": "gemini",
        "planner_model": "gemini-2.5-flash-lite",
        "vision_backend": "openai",
        "vision_model": "gpt-4o",
        "judge_backend": "gemini",
    },
}

# Fallback plan used when the planner fails entirely
_FALLBACK_PLAN = {
    "steps": [
        "Identify the chart type and read the title",
        "Locate axes labels, legend entries, and series names",
        "Extract the data values relevant to the question",
        "Check whether the question is answerable from the chart",
    ],
    "expected_answer_type": "string",
    "question_type": "standard",
    "answerability_check": "uncertain",
    "hints": [],
}


# ---------------------------------------------------------------------------
# Per-sample processing
# ---------------------------------------------------------------------------


def process_sample(  # noqa: PLR0915
    sample: PerceivedSample,
    planner: PlannerAgent,
    vision_agent: VisionAgent,
    config: dict,
    run_id: str,
    out_dir: str,
    lf_client=None,
    verifier_agent: Optional[VerifierAgent] = None,
    ocr_tool: Optional[OcrReaderTool] = None,
) -> str:
    """
    Execute the multi-stage evaluation pipeline for a single sample.

    Coordinates the planner, optional OCR tool, vision agent, and
    optional verifier to produce a Model Evaluation Packet (MEP).

    Parameters
    ----------
    sample : PerceivedSample
        The data sample containing the question and chart image.
    planner : PlannerAgent
        The agent responsible for generating the inspection plan.
    vision_agent : VisionAgent
        The agent that performs visual analysis.
    config : dict
        Configuration dictionary for backends and models.
    run_id : str
        Unique identifier for the current evaluation run.
    out_dir : str
        Directory where the resulting MEP JSON should be saved.
    langfuse_client : object, optional
        The Langfuse client for tracing and observability.
    verifier_agent : VerifierAgent, optional
        The agent for pass 2.5 verification.
    ocr_tool : OcrReaderTool, optional
        Tool for pre-extracting text from the chart.

    Returns
    -------
    str
        The absolute path to the written MEP file.
    """
    config_name = f"{config['planner_backend']}_{config['vision_backend']}"
    run_start = iso_now()
    errors: list = []

    with sample_trace(
        lf_client,
        sample_id=sample.sample_id,
        question=sample.question,
        expected_output=sample.expected_output,
        question_type=sample.question_type.value,
        config_name=config_name,
        run_id=run_id,
    ) as lf_trace:
        lf_trace_id = getattr(lf_trace, "id", None)

        # ---- Planner ----
        plan_prompt = ""
        plan_parsed: dict = {}
        plan_parse_error = True
        plan_raw = ""
        plan_ms = 0.0

        try:
            with timed() as pt:
                plan_prompt, plan_parsed, plan_parse_error, plan_raw = planner.run(sample, lf_trace=lf_trace)

            plan_ms = pt.elapsed_ms
        except Exception as exc:
            errors.append(f"planner_error: {exc}")
            plan_parsed = dict(_FALLBACK_PLAN)
            plan_parsed["question_type"] = sample.question_type.value
            plan_parse_error = True
            traceback.print_exc()

        # ---- OCR pre-read (optional) ----
        ocr_parsed: dict = {}
        ocr_raw = ""
        ocr_parse_error = False
        ocr_traces: list = []
        ocr_ms = 0.0

        if ocr_tool is not None:
            try:
                ocr_tool.lf_trace = lf_trace
                with timed() as ot:
                    ocr_raw = ocr_tool._run(sample.image_path)
                ocr_ms = ot.elapsed_ms
                ocr_traces = ocr_tool.pop_traces()
                ocr_parsed, ocr_ok = parse_strict(ocr_raw, required_keys=["chart_type", "title"])
                ocr_parse_error = not ocr_ok
                if not ocr_parsed:
                    ocr_parsed = {}
            except Exception as exc:
                errors.append(f"ocr_error: {exc}")
                ocr_parse_error = True
                traceback.print_exc()

        # ---- Vision ----
        vision_prompt = ""
        vision_parsed: dict = {}
        vision_parse_error = True
        vision_raw = ""
        vision_traces: list = []
        vision_ms = 0.0

        try:
            with timed() as vt:
                (
                    vision_prompt,
                    vision_parsed,
                    vision_parse_error,
                    vision_raw,
                    vision_traces,
                ) = vision_agent.run(
                    sample,
                    plan_parsed,
                    lf_trace=lf_trace,
                    ocr_result=ocr_parsed if ocr_parsed else None,
                )
            vision_ms = vt.elapsed_ms
        except Exception as exc:
            errors.append(f"vision_error: {exc}")
            vision_parsed = {"answer": "ERROR", "explanation": str(exc)}
            vision_parse_error = True
            traceback.print_exc()

        # ---- Verifier (Pass 2.5) ----
        verifier_prompt = ""
        verifier_parsed: dict = {}
        verifier_parse_error = False
        verifier_raw = ""
        verifier_ms = 0.0
        verifier_verdict = "skipped"

        if verifier_agent is not None:
            try:
                with timed() as vrt:
                    (
                        verifier_prompt,
                        verifier_parsed,
                        verifier_parse_error,
                        verifier_raw,
                    ) = verifier_agent.run(sample, plan_parsed, vision_parsed, lf_trace=lf_trace)

                verifier_ms = vrt.elapsed_ms
                verifier_verdict = verifier_parsed.get("verdict", "confirmed")
            except Exception as exc:
                errors.append(f"verifier_error: {exc}")
                verifier_parsed = {
                    "verdict": "confirmed",
                    "answer": vision_parsed.get("answer", ""),
                    "reasoning": f"Verifier crashed: {exc}",
                }
                verifier_verdict = "confirmed"
                traceback.print_exc()

        run_end = iso_now()

        # ---- Image ref ----
        image_sha = ""
        if sample.image_path and Path(sample.image_path).exists():
            with contextlib.suppress(Exception):
                image_sha = sha256_file(sample.image_path)

        # ---- Assemble MEP ----
        mep = MEP(
            run_id=run_id,
            config=MEPConfig(
                planner_backend=config["planner_backend"],
                vision_backend=config["vision_backend"],
                judge_backend=config.get("judge_backend", config["planner_backend"]),
                config_name=config_name,
                planner_model=config["planner_model"],
                vision_model=config["vision_model"],
            ),
            sample=MEPSample(
                dataset="ChartQAPro",
                sample_id=sample.sample_id,
                question=sample.question,
                question_type=sample.question_type.value,
                expected_output=sample.expected_output,
                image_ref=ImageRef(path=sample.image_path, sha256=image_sha),
                metadata=sample.metadata,
            ),
            plan=MEPPlan(
                prompt=plan_prompt,
                raw_text=plan_raw,
                parsed=plan_parsed,
                parse_error=plan_parse_error,
            ),
            ocr=MEPOcr(
                raw_text=ocr_raw,
                parsed=ocr_parsed,
                parse_error=ocr_parse_error,
                tool_trace=ocr_traces,
            )
            if ocr_tool is not None
            else None,
            vision=MEPVision(
                prompt=vision_prompt,
                raw_text=vision_raw,
                parsed=vision_parsed,
                parse_error=vision_parse_error,
                tool_trace=vision_traces,
            ),
            verifier=MEPVerifier(
                prompt=verifier_prompt,
                raw_text=verifier_raw,
                parsed=verifier_parsed,
                parse_error=verifier_parse_error,
                verdict=verifier_verdict,
            )
            if verifier_agent is not None
            else None,
            timestamps=MEPTimestamps(
                start=run_start,
                end=run_end,
                planner_ms=plan_ms,
                ocr_ms=ocr_ms,
                vision_ms=vision_ms,
                verifier_ms=verifier_ms,
            ),
            errors=errors,
            lf_trace_id=lf_trace_id,
        )

        # ---- Immediately log available scores to Langfuse ----
        log_trace_scores(
            lf_trace,
            {
                "planner_parse_ok": float(not plan_parse_error),
                "vision_parse_ok": float(not vision_parse_error),
                "has_errors": float(bool(errors)),
            },
        )
        if lf_trace:
            lf_trace.update(output=vision_parsed if vision_parsed else None)

    return write_mep(mep, out_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: PLR0912, PLR0915
    """
    Parse CLI arguments and run the MEP generation pipeline.

    Configures agents, loads the dataset, and manages parallel execution
    of the evaluation pipeline.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(description="Generate MEPs for ChartQAPro")
    parser.add_argument(
        "--dataset",
        default="chartqapro",
        help="Dataset name (currently only chartqapro)",
    )
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument("--n", type=int, default=10, help="Number of samples to process")
    parser.add_argument(
        "--config",
        default="gemini_gemini",
        choices=list(BACKEND_CONFIGS.keys()),
        help="Backend config preset",
    )
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (1 = sequential)")
    parser.add_argument("--out", default="meps/", help="Output directory for MEPs")
    parser.add_argument(
        "--image_dir",
        default="data/chartqapro_images",
        help="Directory to save/load chart images",
    )
    parser.add_argument("--cache_dir", default=None, help="HuggingFace datasets cache dir")
    parser.add_argument(
        "--planner_model",
        default=None,
        help="Override planner model name (e.g. gpt-4o, o3)",
    )
    parser.add_argument(
        "--vision_model",
        default=None,
        help="Override vision model name (e.g. gpt-4o, gemini-1.5-pro)",
    )
    parser.add_argument(
        "--verifier_model",
        default=None,
        help="Override verifier model (defaults to vision_model)",
    )
    parser.add_argument("--no_verifier", action="store_true", help="Skip Pass 2.5 verifier agent")
    parser.add_argument("--no_ocr", action="store_true", help="Skip OCR pre-read step")
    parser.add_argument(
        "--ocr_model",
        default=None,
        help="Override OCR model (defaults to vision_model)",
    )
    args = parser.parse_args()

    config = dict(BACKEND_CONFIGS[args.config])  # copy so we don't mutate the preset
    if args.planner_model:
        config["planner_model"] = args.planner_model
    if args.vision_model:
        config["vision_model"] = args.vision_model
    run_id = str(uuid.uuid4())
    out_dir = str(
        Path(args.out) / f"{config['planner_backend']}_{config['vision_backend']}" / "chartqapro" / args.split
    )
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset  : {args.dataset} split={args.split} n={args.n}")
    samples = load_chartqapro(
        split=args.split,
        n=args.n,
        image_dir=args.image_dir,
        cache_dir=args.cache_dir,
    )
    print(f"Samples loaded   : {len(samples)}")
    print(f"Config           : {args.config}  run_id={run_id}")
    print(f"Output dir       : {out_dir}")
    print(f"Workers          : {args.workers}")

    # Langfuse: register dataset + version prompts at run start (no-ops if unavailable)
    lf_client = get_client()
    if lf_client:
        print("Langfuse         : enabled")
        register_dataset(samples, split=args.split)
        push_prompts()
    else:
        print("Langfuse         : not configured (set LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY to enable)")

    # Build agents once — run() creates fresh Crew/Tool per call so this is thread-safe
    print("Initialising agents …")
    planner = PlannerAgent(backend=config["planner_backend"], model=config["planner_model"])
    vision_agent = VisionAgent(
        agent_backend=config["planner_backend"],
        agent_model=config["planner_model"],
        vision_backend=config["vision_backend"],
        vision_model=config["vision_model"],
    )
    verifier: Optional[VerifierAgent] = None
    if not args.no_verifier:
        verifier_model = args.verifier_model or config["vision_model"]
        verifier = VerifierAgent(backend=config["vision_backend"], model=verifier_model)
        print(f"Verifier         : enabled ({config['vision_backend']} / {verifier_model})")
    else:
        print("Verifier         : disabled (--no_verifier)")

    ocr: Optional[OcrReaderTool] = None
    if not args.no_ocr:
        ocr_model = args.ocr_model or config["vision_model"]
        ocr = OcrReaderTool(backend=config["vision_backend"], model=ocr_model)
        print(f"OCR pre-reader   : enabled ({config['vision_backend']} / {ocr_model})")
    else:
        print("OCR pre-reader   : disabled (--no_ocr)")
    print()

    if args.workers <= 1:
        for i, sample in enumerate(samples, 1):
            print(f"[{i}/{len(samples)}] {sample.sample_id} …", end=" ", flush=True)
            try:
                path = process_sample(
                    sample,
                    planner,
                    vision_agent,
                    config,
                    run_id,
                    out_dir,
                    lf_client,
                    verifier,
                    ocr,
                )
                print(f"OK → {path}")
            except Exception as exc:
                print(f"ERROR: {exc}")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            future_to_sample = {
                pool.submit(
                    process_sample,
                    s,
                    planner,
                    vision_agent,
                    config,
                    run_id,
                    out_dir,
                    lf_client,
                    verifier,
                    ocr,
                ): s
                for s in samples
            }
            for done, future in enumerate(as_completed(future_to_sample), 1):
                s = future_to_sample[future]
                try:
                    path = future.result()
                    print(f"[{done}/{len(samples)}] {s.sample_id} → {path}")
                except Exception as exc:
                    print(f"[{done}/{len(samples)}] {s.sample_id} ERROR: {exc}")

    print(f"\nDone. MEPs written to: {out_dir}")


if __name__ == "__main__":
    main()
