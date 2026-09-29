import json
import os
from unittest.mock import patch

import pytest

from pico.models import FakeModelClient
from pico.run_store import RunStore
from pico.runtime import Pico, SessionStore
from pico.trajectory_dataset import (
    SCHEMA_VERSION,
    TrajectoryExportError,
    export_trajectory,
    write_trajectory_jsonl,
)
from pico.workspace import WorkspaceContext


def run_fixture(root, outputs, request="Inspect fixture"):
    root.mkdir()
    (root / "README.md").write_text("fixture\n", encoding="utf-8")
    agent = Pico(
        model_client=FakeModelClient(outputs),
        workspace=WorkspaceContext.build(root),
        session_store=SessionStore(root / ".pico" / "sessions"),
        run_store=RunStore(root / ".pico" / "runs"),
        approval_policy="auto",
    )
    agent.ask(request)
    return agent.run_store.trace_path(agent.current_task_state)


def benchmark_row(passed=True):
    return {
        "run_id": "random-run-id", "status": "pass" if passed else "fail",
        "passed": passed, "verifier_passed": passed,
        "verifier_exit_code": 0 if passed else 1,
        "failure_category": None if passed else "verifier_failed",
        "verifier": "don't copy this secret sk-test-secret-123",
        "verifier_stdout": "don't copy this secret sk-test-secret-123",
    }


def test_fake_model_tool_transition_and_benchmark_label_are_byte_stable(tmp_path):
    outputs = [
        '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":1}}</tool>',
        "<final>Done.</final>",
    ]
    first = run_fixture(tmp_path / "first", outputs)
    second = run_fixture(tmp_path / "second", outputs)
    first_output, second_output = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    write_trajectory_jsonl(first_output, [first], benchmark_rows=[benchmark_row()])
    write_trajectory_jsonl(second_output, [second], benchmark_rows=[benchmark_row()])
    assert first_output.read_bytes() == second_output.read_bytes()
    record = json.loads(first_output.read_text(encoding="utf-8"))
    assert record["schema_version"] == SCHEMA_VERSION
    assert "provenance" not in record
    tool, final = record["transitions"]
    assert [tool["step"], final["step"]] == [1, 2]
    assert tool["action"] == {"kind": "tool", "name": "read_file", "args": {"path": "README.md", "start": 1, "end": 1}}
    assert "1: fixture" in tool["observation"]["result"]
    assert tool["terminal_outcome"] is None
    assert final["state"]["previous_observation"] == tool["observation"]
    assert final["terminal_outcome"]["benchmark"]["passed"] is True
    assert "sk-test-secret-123" not in first_output.read_text(encoding="utf-8")
    assert "run_id" not in first_output.read_text(encoding="utf-8")
    assert "created_at" not in first_output.read_text(encoding="utf-8")
    assert "checkpoint_id" not in first_output.read_text(encoding="utf-8")


def test_fake_model_retry_and_failed_verifier_label(tmp_path):
    trace = run_fixture(tmp_path / "retry", ["<tool>not-json</tool>", "<final>Done.</final>"])
    record = export_trajectory(trace, benchmark_row(passed=False))
    retry, final = record["transitions"]
    assert retry["action"] == {"kind": "retry"}
    assert retry["observation"] is None
    assert final["terminal_outcome"]["benchmark"] == {
        "status": "fail", "passed": False, "verifier_passed": False,
        "verifier_exit_code": 1, "failure_category": "verifier_failed",
    }


@pytest.mark.parametrize("mutation,expected", [
    (lambda events: events.pop(next(i for i, e in enumerate(events) if e["event"] == "model_parsed")), "unexpected tool_executed"),
    (lambda events: events.pop(next(i for i, e in enumerate(events) if e["event"] == "tool_executed")), "unexpected checkpoint_created"),
    (lambda events: events.pop(next(i for i, e in enumerate(events) if e["event"] == "checkpoint_created" and e["trigger"] == "tool_executed")), "missing tool checkpoint"),
    (lambda events: events.pop(next(i for i, e in enumerate(events) if e["event"] == "checkpoint_created" and e["trigger"] == "run_finished")), "missing checkpoint_created"),
    (lambda events: events.__setitem__(slice(2, 4), list(reversed(events[2:4]))), "unexpected model_parsed"),
    (lambda events: events.pop(), "incomplete trace"),
])
def test_missing_or_out_of_order_events_fail_explicitly(tmp_path, mutation, expected):
    trace = run_fixture(tmp_path / "case", [
        '<tool>{"name":"read_file","args":{"path":"README.md"}}</tool>', "<final>Done.</final>",
    ])
    events = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    mutation(events)
    trace.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    with pytest.raises(TrajectoryExportError, match=expected):
        export_trajectory(trace)


def test_secret_redaction_is_shared_with_runtime_and_exporter(tmp_path):
    secret = "sk-test-secret-123"
    with patch.dict(os.environ, {"OPENAI_API_KEY": secret}):
        trace = run_fixture(tmp_path / "secret", [
            '<tool>{"name":"read_file","args":{"path":"README.md"}}</tool>',
            f"<final>Handled {secret}.</final>",
        ], request=f"Inspect {secret}")
        output = tmp_path / "secret.jsonl"
        write_trajectory_jsonl(output, [trace], benchmark_rows=[benchmark_row()])
    assert secret not in output.read_text(encoding="utf-8")
    assert "<redacted>" in output.read_text(encoding="utf-8")
    assert secret not in trace.read_text(encoding="utf-8")


def test_provenance_opt_in_and_benchmark_inconsistency_fails(tmp_path):
    trace = run_fixture(tmp_path / "case", ["<final>Done.</final>"])
    record = export_trajectory(trace, include_provenance=True)
    assert record["provenance"]["run_id"] == trace.parent.name
    assert record["provenance"]["checkpoints"]
    row = benchmark_row()
    row["passed"] = False
    with pytest.raises(TrajectoryExportError, match="disagree"):
        export_trajectory(trace, row)
    original = trace.read_bytes()
    with pytest.raises(ValueError, match="cannot overwrite"):
        write_trajectory_jsonl(trace, [trace])
    assert trace.read_bytes() == original
