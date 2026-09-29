"""Read-only conversion of a completed trace into TrajectoryDatasetV1 JSONL.

One JSONL record represents one run; transitions represent model attempts. The
trace does not contain the full prompt or malformed model text, so neither is
reconstructed here. The tool observation is the already clipped trace result.
"""

import argparse
import json
import os
from pathlib import Path

from .redaction import redact_artifact
from .runtime import Pico

SCHEMA_VERSION = "TrajectoryDatasetV1"
_TOOL_OBSERVATION_FIELDS = (
    "result", "tool_status", "tool_error_code", "security_event_type",
    "risk_level", "read_only", "affected_paths", "workspace_changed", "diff_summary",
)
_PROMPT_FIELDS = (
    "resume_status", "budget_reductions", "workspace_changed", "prefix_changed",
    "stale_summary_invalidations", "stale_paths", "runtime_identity_mismatch_fields",
)
_DYNAMIC_KEYS = frozenset((
    "created_at", "run_id", "task_id", "checkpoint_id", "duration_ms",
    "run_duration_ms", "workspace_fingerprint", "prefix_hash", "prompt_cache_key",
    "built_at", "timestamp",
))


class TrajectoryExportError(ValueError):
    """The trace cannot be converted without guessing about missing events."""


def _stable(value):
    if isinstance(value, dict):
        return {key: _stable(item) for key, item in value.items() if key not in _DYNAMIC_KEYS}
    if isinstance(value, list):
        return [_stable(item) for item in value]
    return value


def _required(event, key, line):
    if key not in event:
        raise TrajectoryExportError(f"line {line}: {event['event']} missing {key}")
    return event[key]


def _label(row):
    if row is None:
        return None
    if not isinstance(row, dict):
        raise TrajectoryExportError("benchmark row must be an object")
    if type(row.get("passed")) is not bool or row.get("status") not in ("pass", "fail"):
        raise TrajectoryExportError("benchmark row requires boolean passed and status pass/fail")
    if row["passed"] != (row["status"] == "pass"):
        raise TrajectoryExportError("benchmark row passed/status disagree")
    if type(row.get("verifier_passed")) is not bool or type(row.get("verifier_exit_code")) is not int:
        raise TrajectoryExportError("benchmark row requires verifier_passed and integer verifier_exit_code")
    if row["verifier_passed"] != (row["verifier_exit_code"] == 0):
        raise TrajectoryExportError("benchmark row verifier fields disagree")
    return {
        "status": row["status"],
        "passed": row["passed"],
        "verifier_passed": row["verifier_passed"],
        "verifier_exit_code": row["verifier_exit_code"],
        "failure_category": row.get("failure_category"),
    }


def export_trajectory(trace_path, benchmark_row=None, *, include_provenance=False, secret_env_names=()):
    """Read a trace and return a normalized record without touching run artifacts.

    The benchmark row is the output row of BenchmarkEvaluator.run_task(), not
    the task definition. Provenance is opt-in because its IDs/timestamps are
    intentionally excluded from byte-stable dataset records.
    """
    trace_path = Path(trace_path)
    names = {str(name).upper() for name in secret_env_names}

    def is_secret_name(name):
        return str(name).upper() in names or Pico.looks_sensitive_env_name(name)

    secrets = [value for name, value in os.environ.items() if value and is_secret_name(name)]

    def clean(value):
        return _stable(redact_artifact(value, is_secret_name=is_secret_name, secret_values=secrets))

    events = []
    for line, raw in enumerate(trace_path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            event = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise TrajectoryExportError(f"line {line}: invalid JSON: {exc.msg}") from exc
        if not isinstance(event, dict) or not isinstance(event.get("event"), str):
            raise TrajectoryExportError(f"line {line}: event must be an object with event name")
        events.append((line, event))
    if not events:
        raise TrajectoryExportError("empty trace")
    line, started = events[0]
    if started["event"] != "run_started":
        raise TrajectoryExportError(f"line {line}: expected run_started, got {started['event']}")
    request = clean(_required(started, "user_request", line))
    if not isinstance(request, str):
        raise TrajectoryExportError(f"line {line}: user_request must be a string")

    transitions = []
    phase = "prompt"
    prompt = None
    previous_observation = None
    parsed_kind = None
    provenance = {"task_id": started.get("task_id"), "started_at": started.get("created_at")}
    checkpoints = []
    tool_checkpoint_pending = False
    terminal_checkpoint = False
    terminal_trigger = None

    for line, event in events[1:]:
        name = event["event"]
        if name == "checkpoint_created" and phase in ("requested", "next", "final"):
            trigger = _required(event, "trigger", line)
            _required(event, "checkpoint_id", line)
            if phase == "next" and trigger == "tool_executed" and tool_checkpoint_pending:
                tool_checkpoint_pending = False
            elif phase in ("next", "final") and trigger != "tool_executed":
                if tool_checkpoint_pending or terminal_checkpoint:
                    raise TrajectoryExportError(f"line {line}: unexpected terminal checkpoint")
                terminal_checkpoint = True
                terminal_trigger = trigger
            elif phase == "requested" and trigger in ("freshness_mismatch", "workspace_mismatch", "context_reduction"):
                pass
            else:
                raise TrajectoryExportError(f"line {line}: unexpected checkpoint trigger {trigger!r}")
            checkpoints.append({"checkpoint_id": event.get("checkpoint_id"), "created_at": event.get("created_at")})
            continue
        if name == "runtime_identity_mismatch" and phase == "requested":
            continue
        if phase in ("prompt", "next") and name == "prompt_built":
            if tool_checkpoint_pending or terminal_checkpoint:
                raise TrajectoryExportError(f"line {line}: missing tool checkpoint or unexpected terminal checkpoint")
            metadata = _required(event, "prompt_metadata", line)
            if not isinstance(metadata, dict):
                raise TrajectoryExportError(f"line {line}: prompt_metadata must be an object")
            prompt = clean({key: metadata[key] for key in _PROMPT_FIELDS if key in metadata})
            phase = "requested"
            continue
        if phase == "requested" and name == "model_requested":
            attempt = _required(event, "attempts", line)
            tool_steps = _required(event, "tool_steps", line)
            if type(attempt) is not int or attempt != len(transitions) + 1:
                raise TrajectoryExportError(f"line {line}: model_requested attempts out of sequence")
            if type(tool_steps) is not int or tool_steps != sum(t["action"]["kind"] == "tool" for t in transitions):
                raise TrajectoryExportError(f"line {line}: model_requested tool_steps out of sequence")
            state = {
                "user_request": request,
                "attempt": attempt,
                "tool_steps": tool_steps,
                "prompt_metadata": prompt,
                "previous_observation": previous_observation,
            }
            phase = "parsed"
            continue
        if phase == "parsed" and name == "model_parsed":
            parsed_kind = _required(event, "kind", line)
            if parsed_kind not in ("tool", "retry", "final"):
                raise TrajectoryExportError(f"line {line}: unsupported model_parsed kind {parsed_kind!r}")
            transition = {
                "step": len(transitions) + 1,
                "state": state,
                "action": {"kind": parsed_kind},
                "observation": None,
                "terminal_outcome": None,
            }
            transitions.append(transition)
            phase = "tool" if parsed_kind == "tool" else ("final" if parsed_kind == "final" else "next")
            continue
        if phase == "tool" and name == "tool_executed":
            tool_name = _required(event, "name", line)
            args = _required(event, "args", line)
            result = _required(event, "result", line)
            if not isinstance(tool_name, str) or not isinstance(args, dict) or not isinstance(result, str):
                raise TrajectoryExportError(f"line {line}: invalid tool_executed name/args/result")
            observation = clean({key: event[key] for key in _TOOL_OBSERVATION_FIELDS if key in event})
            transitions[-1]["action"].update({"name": clean(tool_name), "args": clean(args)})
            transitions[-1]["observation"] = observation
            previous_observation = observation
            tool_checkpoint_pending = True
            phase = "next"
            continue
        if phase in ("next", "final") and name == "run_finished":
            if tool_checkpoint_pending or not terminal_checkpoint:
                raise TrajectoryExportError(f"line {line}: missing checkpoint_created before run_finished")
            status = _required(event, "status", line)
            stop_reason = _required(event, "stop_reason", line)
            final_answer = _required(event, "final_answer", line)
            if not all(isinstance(item, str) for item in (status, stop_reason, final_answer)):
                raise TrajectoryExportError(f"line {line}: invalid run_finished fields")
            if (phase == "final") != (stop_reason == "final_answer_returned"):
                raise TrajectoryExportError(f"line {line}: final action and stop_reason disagree")
            if terminal_trigger not in ("run_finished", stop_reason):
                raise TrajectoryExportError(f"line {line}: terminal checkpoint and stop_reason disagree")
            if phase == "final":
                transitions[-1]["action"]["final_answer"] = clean(final_answer)
            label = _label(benchmark_row)
            transitions[-1]["terminal_outcome"] = {
                "status": clean(status),
                "stop_reason": clean(stop_reason),
                "final_answer": clean(final_answer),
                "benchmark": clean(label) if label is not None else None,
            }
            provenance["finished_at"] = event.get("created_at")
            phase = "done"
            continue
        raise TrajectoryExportError(f"line {line}: unexpected {name} while expecting {phase}")

    if phase != "done" or not transitions:
        raise TrajectoryExportError(f"incomplete trace: expected run_finished after complete step (phase={phase})")
    record = {"schema_version": SCHEMA_VERSION, "transitions": transitions}
    if include_provenance:
        provenance["run_id"] = benchmark_row.get("run_id") if benchmark_row else trace_path.parent.name
        provenance["checkpoints"] = checkpoints
        record["provenance"] = redact_artifact(
            provenance, is_secret_name=is_secret_name, secret_values=secrets
        )
    return record


def write_trajectory_jsonl(output_path, traces, *, benchmark_rows=None, include_provenance=False):
    """Write one canonical JSON line per trace, in caller-specified trace order."""
    traces = list(traces)
    output_path = Path(output_path)
    if any(output_path.resolve() == Path(trace).resolve() for trace in traces):
        raise ValueError("output path cannot overwrite an input trace")
    rows = [None] * len(traces) if benchmark_rows is None else list(benchmark_rows)
    if len(rows) != len(traces):
        raise ValueError("benchmark_rows must match traces in length and order")
    # Validate all traces before opening the output file.
    records = [
        export_trajectory(trace, row, include_provenance=include_provenance)
        for trace, row in zip(traces, rows)
    ]
    payload = "".join(json.dumps(record, sort_keys=True, ensure_ascii=True, separators=(",", ":")) + "\n" for record in records)
    output_path.write_bytes(payload.encode("utf-8"))
    return output_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export completed Pico traces as TrajectoryDatasetV1 JSONL")
    parser.add_argument("trace", nargs="+", type=Path, help="trace.jsonl paths, in output order")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--benchmark-rows", type=Path, help="JSON file with a rows array in the same order as traces")
    parser.add_argument("--include-provenance", action="store_true")
    args = parser.parse_args(argv)
    rows = None
    if args.benchmark_rows:
        data = json.loads(args.benchmark_rows.read_text(encoding="utf-8"))
        rows = data["rows"]
    write_trajectory_jsonl(args.output, args.trace, benchmark_rows=rows, include_provenance=args.include_provenance)


if __name__ == "__main__":
    main()
