import numpy as np

from phishdet.dataset import (
    Record,
    cap_groups,
    dedupe_exact,
    looks_english,
    near_duplicate_groups,
    stratified_group_split,
    time_split,
)


def _rec(i, text, label, year=2020):
    return Record(id=str(i), source="t", subset="t", label=label, year=year, text=text, raw_text="", parsed_text="", has_html=False, origin=str(i))


def test_language_check():
    assert looks_english("Please verify your account to keep it active, we have noticed a problem with the login.")
    assert not looks_english("Estimado cliente, por favor actualice los datos de su cuenta para que podamos continuar con el servicio.")
    assert not looks_english("您的帐户安全是我们的首要任务。最近我们已经要求所有用户重新验证他们的帐户")
    # too short to tell -> kept
    assert looks_english("ok thanks")


def test_exact_dedupe_ignores_case_and_punctuation():
    recs = [_rec(1, "Verify your account!", 1), _rec(2, "verify your account", 1), _rec(3, "Lunch at noon?", 0)]
    kept, dropped = dedupe_exact(recs)
    assert [r.id for r in kept] == ["1", "3"]
    assert sum(dropped.values()) == 1


def test_exact_duplicates_with_conflicting_labels_are_all_dropped():
    recs = [_rec(1, "same text here", 1), _rec(2, "same text here", 0), _rec(3, "other", 0)]
    kept, dropped = dedupe_exact(recs)
    assert [r.id for r in kept] == ["3"]
    assert dropped["label_conflict_exact"] == 2


def test_near_duplicates_grouped():
    base = "your mailbox has exceeded its storage limit please click the link below to verify your account now"
    texts = [
        base + " ref 1001",
        base + " ref 2002",
        "the quarterly gas trading report is attached please review before the meeting on thursday afternoon",
        base.replace("mailbox", "inbox") + " ref 3003",
    ]
    groups = near_duplicate_groups(texts, threshold=0.7, chunk=2)
    assert groups[0] == groups[1]
    assert groups[2] != groups[0]
    assert groups[3] == groups[0]


def test_cap_groups_limits_members_and_drops_mixed_groups():
    recs = [_rec(i, "x", 1) for i in range(5)] + [_rec(5, "y", 1), _rec(6, "y", 0)]
    groups = np.array([0, 0, 0, 0, 0, 1, 1])
    keep, dropped = cap_groups(recs, groups, max_per_group=2)
    assert len(keep) == 2 and all(i < 5 for i in keep)
    assert dropped["label_conflict_near_duplicate"] == 2


def test_stratified_group_split_keeps_groups_together():
    rng = np.random.default_rng(0)
    labels = np.array([0] * 300 + [1] * 300)
    groups = rng.integers(0, 200, size=600)
    split = stratified_group_split(labels, groups, seed=1)
    for g in np.unique(groups):
        assert len(set(split[groups == g])) == 1
    frac_test = np.mean(split == "test")
    assert 0.12 < frac_test < 0.28
    assert set(split) == {"train", "val", "test"}


def test_time_split_moves_late_phishing_to_test_and_blocks_seen_campaigns():
    labels = np.array([1, 1, 1, 1, 0, 0])
    years = [2006, 2018, 2022, 2023, 2002, 2002]
    groups = np.array([0, 1, 2, 1, 3, 4])  # the 2023 email shares a group with a 2018 one
    base = np.array(["train", "train", "train", "train", "train", "test"], dtype=object)
    split = time_split(labels, years, groups, base, cutoff=2020)
    assert split[2] == "test"
    assert split[3] == "unused"
    assert split[0] in ("train", "val") and split[1] in ("train", "val")
    assert split[4] == "train" and split[5] == "test"
