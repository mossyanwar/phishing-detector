"""Run scripts/run_llm_eval.py end to end on a tiny synthetic dataset with the fake provider."""

import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from phishdet.dataset import Record

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_llm_eval.py"


@pytest.fixture()
def script(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("run_llm_eval", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(mod, "SAMPLE", tmp_path / "data" / "llm_sample_ids.json")
    (tmp_path / "data" / "processed").mkdir(parents=True)
    recs = []
    for i in range(6):
        phish = i % 2 == 0
        text = ("Subject: account suspended\n\nVerify your password at url:x.example, click now" if phish
                else "Subject: Re: notes\n\nThanks, see you at the meeting")
        recs.append(Record(id=f"id{i}", source="t", subset="t", label=int(phish), year=2020, text=text,
                           raw_text="", parsed_text="", has_html=False, origin=str(i),
                           meta={"group": i, "split": "test", "time_split": "test"}))
    with (tmp_path / "data" / "processed" / "emails.jsonl").open("w") as f:
        for r in recs:
            f.write(json.dumps(asdict(r)) + "\n")
    return mod


def test_fake_run_writes_local_outputs_only(script, tmp_path, capsys):
    assert script.main(["--make-sample", "--n-per-class", "2"]) == 0
    assert script.main(["--provider", "fake", "--mode", "few-shot"]) == 0
    out = json.loads((tmp_path / "results" / "llm_runs" / "fake__fake-rules__few-shot.json").read_text())
    assert out["n"] == 4 and out["n_invalid"] == 0
    assert out["recall"] == 1.0 and out["fpr"] == 0.0
    assert not (tmp_path / "results" / "llm").exists()  # fake numbers never land next to real results
    lines = (tmp_path / "results" / "llm_runs" / "fake__fake-rules__few-shot.jsonl").read_text().splitlines()
    assert len(lines) == 4 and "reason" in json.loads(lines[0])


def test_paid_provider_needs_yes_and_never_builds_a_client(script, monkeypatch, capsys):
    script.main(["--make-sample", "--n-per-class", "2"])

    def boom(*a, **k):
        raise AssertionError("provider must not be constructed without --yes")

    monkeypatch.setattr(script, "make_provider", boom)
    assert script.main(["--provider", "anthropic", "--mode", "zero-shot"]) == 0
    out = capsys.readouterr().out
    assert "estimated cost $" in out and "add --yes" in out
