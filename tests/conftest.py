import numpy as np
import pytest

from phishdet.baseline import TfidfModel

PHISH = [
    "Subject: verify your account\n\nDear customer, your account is suspended. Verify your account at url:secure-bank.example now.",
    "Subject: mailbox full\n\nYour mailbox is full. Click url:ip-address to verify your email account or it will be closed.",
    "Subject: security alert\n\nWe detected unusual sign in activity. Confirm your password at url:paypal-login.example today.",
    "Subject: invoice overdue\n\nYour payment failed, update your card details at url:billing-update.example to avoid suspension.",
    "Subject: account locked\n\nDear user, your account has been locked. Verify your identity at url:account-check.example.",
    "Subject: urgent notice\n\nYour email will be deactivated. Click here url:webmail-verify.example to keep your account.",
]
LEGIT = [
    "Subject: Re: meeting notes\n\nThanks, I moved the review to Thursday. Let me know if that works for the team.",
    "Subject: Re: kernel patch\n\nThe patch builds on my machine, but the driver still fails to load after reboot.",
    "Subject: lunch\n\nAre we still on for lunch at noon? I can book a table near the office.",
    "Subject: Re: gas contract\n\nI reviewed the draft and left comments on section two. Call me when you have a minute.",
    "Subject: Re: release plan\n\nI think we should ship the fix first and leave the refactor for the next release.",
    "Subject: draft report\n\nAttached is the draft. The numbers in the second table still need checking.",
]


@pytest.fixture(scope="session")
def toy_texts():
    texts = PHISH + LEGIT
    labels = np.array([1] * len(PHISH) + [0] * len(LEGIT))
    return texts, labels


@pytest.fixture(scope="session")
def toy_model(toy_texts):
    texts, labels = toy_texts
    return TfidfModel.train(texts, labels, C=10.0, min_df=1)
