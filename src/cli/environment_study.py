"""Unified command surface for CommerceWorld environment studies."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

from experiments.catalog_feasibility import (
    CATALOG_FEASIBILITY_REPORT_SCHEMA,
    CatalogFeasibilityError,
    CatalogFeasibilityReportV1,
    build_catalog_feasibility_report,
    verify_catalog_feasibility_report,
)
from experiments.data_environment_study import (
    DataEnvironmentStudyError,
    run_persisted_data_study,
)
from experiments.e1_persistence_study import (
    E1PersistenceStudyError,
    run_persisted_e1_study,
)
from experiments.environment_study import write_json_artifact
from experiments.environment_integrity_study import (
    EnvironmentIntegrityStudyError,
    run_persisted_integrity_study,
)
from experiments.multiagent_preflight import (
    MultiAgentPreflightError,
    run_persisted_multiagent_preflight,
)
from experiments.real_catalog_multiagent import (
    RealCatalogMultiAgentError,
    run_persisted_real_catalog_multiagent,
)
from experiments.multiagent_openrouter import (
    DEFAULT_OPENROUTER_MODEL,
    MULTIAGENT_LLM_TRACE_REPORT_SCHEMA,
    MultiAgentLLMTraceReportV1,
    MultiAgentOpenRouterError,
    PaidAuthorizationRequired,
    build_multiagent_openrouter_contract,
    fetch_openrouter_model_snapshot,
    run_persisted_multiagent_llm_trace,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="acworld-environment-study")
    commands = parser.add_subparsers(dest="command", required=True)
    feasibility = commands.add_parser(
        "feasibility",
        help="build the sanitized real-catalog feasibility report",
    )
    feasibility.add_argument("--out", type=Path, required=True)
    data_run = commands.add_parser(
        "data-run",
        help="run CWENV-DATA-01 through Agent, Platform, and World",
    )
    data_run.add_argument("--out-root", type=Path, required=True)
    data_run.add_argument("--artifacts-dir", type=Path, required=True)
    data_run.add_argument("--report-out", type=Path, required=True)
    data_run.add_argument(
        "--multiagent",
        action="store_true",
        help="run the local-only real-CSV 5x5 Agent market instead of CWENV-DATA-01",
    )
    m2m_preflight = commands.add_parser(
        "m2m-preflight",
        help="run the egress-free 2x2 and 5x5 shared-market preflight",
    )
    m2m_preflight.add_argument("--out-root", type=Path, required=True)
    m2m_preflight.add_argument("--artifacts-dir", type=Path, required=True)
    m2m_preflight.add_argument("--report-out", type=Path, required=True)
    m2m_run = commands.add_parser(
        "m2m-run",
        help="run the single frozen 5x5 OpenRouter trace behind a paid gate",
    )
    m2m_run.add_argument("--out-root", type=Path, required=True)
    m2m_run.add_argument("--artifacts-dir", type=Path, required=True)
    m2m_run.add_argument("--report-out", type=Path, required=True)
    m2m_run.add_argument(
        "--model-id",
        choices=(DEFAULT_OPENROUTER_MODEL,),
        default=DEFAULT_OPENROUTER_MODEL,
    )
    m2m_run.add_argument("--api-key-file", type=Path)
    m2m_run.add_argument(
        "--allow-paid",
        action="store_true",
        help="fresh explicit authorization for up to $3.00 of provider charges",
    )
    e1_run = commands.add_parser(
        "e1-run",
        help="run the three-Episode DatabaseWorld E1 persistence study",
    )
    e1_run.add_argument("--out-root", type=Path, required=True)
    e1_run.add_argument("--artifacts-dir", type=Path, required=True)
    e1_run.add_argument("--report-out", type=Path, required=True)
    integrity_run = commands.add_parser(
        "integrity-run",
        help="refresh three extension cases and run ten model-free integrity probes",
    )
    integrity_run.add_argument("--out-root", type=Path, required=True)
    integrity_run.add_argument("--artifacts-dir", type=Path, required=True)
    integrity_run.add_argument("--report-out", type=Path, required=True)
    verify = commands.add_parser("verify", help="recompute and verify a study artifact")
    verify.add_argument("--artifact", type=Path, required=True)
    return parser


def _emit(value: object, *, error: bool = False) -> None:
    print(
        json.dumps(value, ensure_ascii=False, sort_keys=True),
        file=sys.stderr if error else sys.stdout,
    )


def _read_openrouter_api_key(path: Path | None) -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        source = path or (_REPO_ROOT / "openrouter_APIkey.txt")
        if source.is_file():
            key = source.read_text(encoding="utf-8").strip()
    if not key:
        raise MultiAgentOpenRouterError(
            "no OpenRouter API key (set OPENROUTER_API_KEY or --api-key-file)"
        )
    return key


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "feasibility":
            report = build_catalog_feasibility_report()
            write_json_artifact(report.to_dict(), args.out)
            _emit(
                {
                    "schema_version": CATALOG_FEASIBILITY_REPORT_SCHEMA,
                    "report_id": report.report_id,
                    "path": str(args.out),
                    "selected_merchant_ids": list(report.selected_merchant_ids),
                    "real_csv_5x5_status": report.real_csv_5x5_status,
                    "paper_result": False,
                }
            )
            return 0

        if args.command == "m2m-preflight":
            preflight_result = run_persisted_multiagent_preflight(
                out_root=args.out_root,
                artifacts_dir=args.artifacts_dir,
            )
            write_json_artifact(preflight_result.report.to_dict(), args.report_out)
            _emit(
                {
                    "schema_version": preflight_result.report.schema_version,
                    "study_id": preflight_result.report.study_id,
                    "contract_id": preflight_result.report.contract_id,
                    "valid": preflight_result.report.valid,
                    "path": str(args.report_out),
                    "provider_calls": preflight_result.report.provider_calls,
                    "paper_result": False,
                }
            )
            return 0

        if args.command == "m2m-run":
            # Public GET only.  Print the exact model and worst-case price
            # before the API key is read or any paid endpoint is reachable.
            preview_snapshot = fetch_openrouter_model_snapshot()
            preview_contract, _ = build_multiagent_openrouter_contract(
                preview_snapshot,
                repo_root=_REPO_ROOT,
            )
            _emit(
                {
                    "schema_version": preview_contract.schema_version,
                    "contract_id": preview_contract.contract_id,
                    "exact_model_id": preview_contract.model.model_id,
                    "canonical_slug": preview_contract.model.canonical_slug,
                    "maximum_billable_attempts": (
                        preview_contract.max_billable_attempts
                    ),
                    "hard_cost_cap_usd": preview_contract.hard_cost_cap_usd,
                    "worst_case_cost_usd": preview_contract.worst_case_cost_usd,
                    "authorization_required": not args.allow_paid,
                    "paid_request_sent": False,
                },
                error=True,
            )
            if not args.allow_paid:
                raise PaidAuthorizationRequired(
                    "review the printed contract and rerun with --allow-paid"
                )
            api_key = _read_openrouter_api_key(args.api_key_file)
            paid_result = run_persisted_multiagent_llm_trace(
                allow_paid=True,
                api_key=api_key,
                out_root=args.out_root,
                artifacts_dir=args.artifacts_dir,
                repo_root=_REPO_ROOT,
            )
            write_json_artifact(paid_result.report.to_dict(), args.report_out)
            _emit(
                {
                    "schema_version": paid_result.report.schema_version,
                    "report_id": paid_result.report.report_id,
                    "contract_id": paid_result.report.contract_id,
                    "valid": paid_result.report.valid,
                    "provider_attempts": paid_result.report.provider_attempts,
                    "total_cost_usd": paid_result.report.total_cost_usd,
                    "actor_count": len(paid_result.report.actor_provider_calls),
                    "path": str(args.report_out),
                    "paper_result": paid_result.report.valid,
                }
            )
            return 0 if paid_result.report.valid else 1

        if args.command == "data-run":
            data_result = (
                run_persisted_real_catalog_multiagent(
                    out_root=args.out_root,
                    artifacts_dir=args.artifacts_dir,
                )
                if args.multiagent
                else run_persisted_data_study(
                    out_root=args.out_root,
                    artifacts_dir=args.artifacts_dir,
                )
            )
            write_json_artifact(data_result.report.to_dict(), args.report_out)
            _emit(
                {
                    "schema_version": data_result.report.schema_version,
                    "study_id": data_result.report.study_id,
                    "contract_id": data_result.report.contract_id,
                    "valid": data_result.report.valid,
                    "path": str(args.report_out),
                    "provider_calls": data_result.report.provider_calls,
                    "actor_count": len(data_result.report.actor_coverage),
                    "paper_result": False,
                }
            )
            return 0

        if args.command == "e1-run":
            e1_result = run_persisted_e1_study(
                out_root=args.out_root,
                artifacts_dir=args.artifacts_dir,
            )
            write_json_artifact(e1_result.report.to_dict(), args.report_out)
            _emit(
                {
                    "schema_version": e1_result.report.schema_version,
                    "study_id": e1_result.report.study_id,
                    "contract_id": e1_result.report.contract_id,
                    "valid": e1_result.report.valid,
                    "episode_count": e1_result.report.data_scope["episode_count"],
                    "provider_calls": e1_result.report.provider_calls,
                    "path": str(args.report_out),
                    "paper_result": e1_result.report.data_scope["paper_result"],
                }
            )
            return 0 if e1_result.report.valid else 1

        if args.command == "integrity-run":
            integrity_result = run_persisted_integrity_study(
                out_root=args.out_root,
                artifacts_dir=args.artifacts_dir,
            )
            write_json_artifact(integrity_result.report.to_dict(), args.report_out)
            _emit(
                {
                    "schema_version": integrity_result.report.schema_version,
                    "study_id": integrity_result.report.study_id,
                    "contract_id": integrity_result.report.contract_id,
                    "valid": integrity_result.report.valid,
                    "extension_case_count": integrity_result.report.diagnostics[
                        "extension_case_count"
                    ],
                    "integrity_case_count": integrity_result.report.diagnostics[
                        "integrity_case_count"
                    ],
                    "provider_calls": integrity_result.report.provider_calls,
                    "path": str(args.report_out),
                    "paper_result": integrity_result.report.data_scope["paper_result"],
                }
            )
            return 0 if integrity_result.report.valid else 1

        raw = json.loads(args.artifact.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise CatalogFeasibilityError("study artifact must be a JSON object")
        if raw.get("schema_version") == CATALOG_FEASIBILITY_REPORT_SCHEMA:
            report = CatalogFeasibilityReportV1.from_dict(raw)
            verify_catalog_feasibility_report(report)
            verified = {
                "schema_version": CATALOG_FEASIBILITY_REPORT_SCHEMA,
                "verified": True,
                "report_id": report.report_id,
                "paper_result": False,
            }
        elif raw.get("schema_version") == MULTIAGENT_LLM_TRACE_REPORT_SCHEMA:
            paid_report = MultiAgentLLMTraceReportV1.from_dict(raw)
            verified = {
                "schema_version": paid_report.schema_version,
                "verified": paid_report.valid,
                "report_id": paid_report.report_id,
                "contract_id": paid_report.contract_id,
                "provider_attempts": paid_report.provider_attempts,
                "paper_result": paid_report.valid,
            }
        else:
            from experiments.environment_study import EnvironmentStudyReportV1

            environment_report = EnvironmentStudyReportV1.from_dict(raw)
            verified = {
                "schema_version": environment_report.schema_version,
                "verified": environment_report.valid,
                "study_id": environment_report.study_id,
                "contract_id": environment_report.contract_id,
                "provider_calls": environment_report.provider_calls,
                "paper_result": bool(
                    environment_report.data_scope.get("paper_result", False)
                ),
            }
        _emit(verified)
        return 0
    except PaidAuthorizationRequired as exc:
        _emit(
            {
                "schema_version": "cwe.environment-study-paid-authorization.v1",
                "authorized": False,
                "paid_request_sent": False,
                "error": str(exc),
            },
            error=True,
        )
        return 2
    except (
        CatalogFeasibilityError,
        DataEnvironmentStudyError,
        E1PersistenceStudyError,
        EnvironmentIntegrityStudyError,
        MultiAgentPreflightError,
        MultiAgentOpenRouterError,
        RealCatalogMultiAgentError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        _emit(
            {
                "schema_version": "cwe.environment-study-cli-error.v1",
                "verified": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
            error=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
