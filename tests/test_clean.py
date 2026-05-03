from phishdet.clean import (
    clean_body,
    clean_email,
    clean_subject,
    normalise_unicode,
    replace_tokens,
    strip_structure,
)


def test_zero_width_characters_are_removed():
    assert normalise_unicode("pa​ss‍word") == "password"


def test_homoglyphs_fixed_only_in_mixed_script_words():
    # Cyrillic 'а' and 'о' inside a Latin word
    assert normalise_unicode("PаyPаl lоgin") == "PayPal login"
    # an all-Cyrillic word is left alone
    assert normalise_unicode("привет") == "привет"


def test_fullwidth_letters_are_folded():
    assert normalise_unicode("Ｐａｙ") == "Pay"


def test_urls_become_host_tokens():
    out = replace_tokens("go to https://www.Secure-Login.example.com/path?a=1 now")
    assert "url:secure-login.example.com" in out
    assert "/path" not in out


def test_ip_urls_and_userinfo_trick():
    assert "url:ip-address" in replace_tokens("http://192.168.10.5/login")
    # the real host of http://paypal.com@evil.example is evil.example
    assert "url:evil.example" in replace_tokens("http://paypal.com@evil.example/x")


def test_emails_years_and_recipient_tokens():
    out = replace_tokens("Dear jose, write to help@bank.example by 2024. Enron staff at monkey.org")
    assert "EMAIL" in out and "help@bank.example" not in out
    assert "YEAR" in out and "2024" not in out
    assert "RECIPIENT" in out and "jose" not in out
    assert out.count("ORGNAME") == 2


def test_san_jose_is_not_a_recipient():
    assert "San Jose" in replace_tokens("offices in San Jose")


def test_lotus_notes_addresses_are_replaced():
    out = replace_tokens("cc: Sara Shackleton/HOU/ECT@ECT, Kay Mann/Corp/Enron@ENRON")
    assert "HOU/ECT" not in out and "Corp/" not in out
    assert "NAME" in out


def test_quoted_reply_and_attribution_removed():
    body = "Sounds good.\n\nOn Tue, 1 Oct 2002, Bob wrote:\n> are we still on?\n> thanks\n"
    out = strip_structure(body)
    assert "Sounds good." in out
    assert "are we still on" not in out
    assert "wrote" not in out


def test_signature_and_list_footer_cut():
    body = (
        "Here is the patch.\nIt fixes the bug.\n\n-- \nIrish Linux Users' Group: ilug@linux.ie\n"
        "http://www.linux.ie/mailman/listinfo/ilug\n"
    )
    out = strip_structure(body)
    assert "patch" in out
    assert "listinfo" not in out and "Irish Linux" not in out

    body2 = (
        "Real content line.\n_______________________________________________\n"
        "Spamassassin-talk mailing list\nhttps://lists.sourceforge.net/lists/listinfo/spamassassin-talk\n"
    )
    out2 = strip_structure(body2)
    assert out2.strip() == "Real content line."


def test_long_tail_after_separator_is_kept():
    # a separator near the top of a phishing email must not delete the message
    lines = ["Dear customer"] + ["_" * 20] + [f"line {i}" for i in range(30)] + ["to unsubscribe click here"]
    out = strip_structure("\n".join(lines))
    assert "line 29" in out


def test_embedded_forward_headers_removed():
    body = "FYI\n---------------------- Forwarded by Kay Mann on 05/01/2001\nFrom: someone\nSubject: x\nThe deal is done."
    out = strip_structure(body)
    assert "Forwarded by" not in out and "From:" not in out
    assert "The deal is done." in out


def test_subject_list_tag_stripped_but_reply_prefix_kept():
    assert clean_subject("[ILUG] kernel panic") == "kernel panic"
    assert clean_subject("Re: [SAtalk] razor") == "Re: razor"
    # multi-word bracket text is real content and stays
    assert clean_subject("[Action Required] verify") == "[Action Required] verify"


def test_clean_email_format_and_truncation():
    text = clean_email("Hi", "word " * 2000, max_chars=100)
    assert text.startswith("Subject: Hi\n\n")
    assert len(text) == 100


def test_clean_body_collapses_whitespace():
    assert clean_body("a  \t b\n\n\n\nc") == "a b\n\nc"


def test_cleaning_is_idempotent_enough():
    once = clean_email("Verify", "Click http://evil.example/x now, jose. 2024")
    body = once.split("\n\n", 1)[1]
    assert clean_body(body) == body


def test_numeric_dates_are_replaced():
    out = replace_tokens("payment due on 9/30/01 or 2024-03-17, call 0800 123 456")
    assert out.count("DATE") == 2
    assert "0800 123 456" in out
