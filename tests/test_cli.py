import json
from email.message import EmailMessage

import pytest

from phishdet.cli import main, score_bytes


def _eml(tmp_path, name, subject, body, html=False):
    msg = EmailMessage()
    msg["From"] = "someone@example.org"
    msg["Subject"] = subject
    msg.set_content(body, subtype="html" if html else "plain")
    path = tmp_path / name
    path.write_bytes(msg.as_bytes())
    return path


@pytest.fixture()
def model_path(toy_model, tmp_path):
    path = tmp_path / "model.joblib"
    toy_model.save(path)
    return path


def test_score_bytes_on_phishing_html(toy_model):
    msg = EmailMessage()
    msg["Subject"] = "verify your account"
    msg.set_content("<p>Your account is suspended. <a href='http://secure-bank.example/x'>Verify</a></p>", subtype="html")
    r = score_bytes(msg.as_bytes(), toy_model, threshold=0.5, top=3)
    assert r["verdict"] == "phishing"
    assert len(r["top_features"]) == 3
    assert set(r["totals"]) == {"word", "char"}


def test_cli_json_output(tmp_path, model_path, capsys):
    phish = _eml(tmp_path, "p.eml", "account locked", "Dear user, verify your account at http://account-check.example/login")
    legit = _eml(tmp_path, "l.eml", "Re: meeting notes", "Thanks, I moved the review to Thursday. Let me know.")
    code = main(["score", str(phish), str(legit), "--model-path", str(model_path), "--threshold", "0.5", "--json"])
    assert code == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert [r["verdict"] for r in lines] == ["phishing", "legitimate"]
    assert lines[0]["file"].endswith("p.eml")
    assert 0.0 <= lines[1]["probability_phishing"] <= 1.0


def test_cli_human_output_shows_reasons(tmp_path, model_path, capsys):
    phish = _eml(tmp_path, "p.eml", "mailbox full", "Your mailbox is full, verify your email account now")
    assert main(["score", str(phish), "--model-path", str(model_path), "--threshold", "0.5", "--top", "4"]) == 0
    out = capsys.readouterr().out
    assert "P(phish):" in out and "PHISHING" in out
    assert "why" in out and "word:" in out


def test_cli_missing_model_is_a_clear_error(tmp_path, capsys):
    eml = _eml(tmp_path, "x.eml", "hi", "hello")
    code = main(["score", str(eml), "--model-path", str(tmp_path / "nope.joblib")])
    assert code == 2
    assert "no trained model" in capsys.readouterr().err


def test_cli_unreadable_file(tmp_path, model_path, capsys):
    code = main(["score", str(tmp_path / "missing.eml"), "--model-path", str(model_path), "--threshold", "0.5"])
    assert code == 2
    assert "cannot read" in capsys.readouterr().err
