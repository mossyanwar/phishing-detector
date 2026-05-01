from email.message import EmailMessage

from phishdet.parsing import html_to_text, parse_message


def _plain(subject="Hello", body="Hi there,\nsee you soon.\n", date="Tue, 01 Oct 2002 10:00:00 +0100"):
    msg = EmailMessage()
    msg["From"] = "alice@example.org"
    msg["To"] = "bob@example.org"
    msg["Subject"] = subject
    if date:
        msg["Date"] = date
    msg.set_content(body)
    return msg.as_bytes()


def test_plain_message_subject_body_and_date():
    p = parse_message(_plain())
    assert p.subject == "Hello"
    assert "see you soon." in p.body
    assert p.date.year == 2002
    assert not p.has_html


def test_headers_other_than_subject_are_not_in_body():
    p = parse_message(_plain())
    assert "alice@example.org" not in p.body
    assert "Date:" not in p.body


def test_html_only_message_is_converted_with_link_targets():
    msg = EmailMessage()
    msg["Subject"] = "Verify your account"
    msg.set_content(
        "<html><head><style>p {color:red}</style></head><body>"
        "<p>Dear customer,</p><p><a href='http://evil.example/login'>Sign in</a> now</p>"
        "<script>alert(1)</script></body></html>",
        subtype="html",
    )
    p = parse_message(msg.as_bytes())
    assert p.has_html
    assert "Dear customer," in p.body
    assert "Sign in http://evil.example/login" in " ".join(p.body.split())
    assert "color:red" not in p.body
    assert "alert(1)" not in p.body


def test_multipart_alternative_prefers_plain_text():
    msg = EmailMessage()
    msg["Subject"] = "Both"
    msg.set_content("plain version")
    msg.add_alternative("<p>html version</p>", subtype="html")
    p = parse_message(msg.as_bytes())
    assert p.body.strip() == "plain version"
    assert p.has_html


def test_attachments_are_ignored():
    msg = EmailMessage()
    msg["Subject"] = "Invoice"
    msg.set_content("see attached")
    msg.add_attachment(b"secret text in a file", maintype="text", subtype="plain", filename="a.txt")
    p = parse_message(msg.as_bytes())
    assert "secret text" not in p.body


def test_quoted_printable_and_charset_decoding():
    raw = (
        b"Subject: =?utf-8?q?Caf=C3=A9?=\r\n"
        b"Content-Type: text/plain; charset=utf-8\r\n"
        b"Content-Transfer-Encoding: quoted-printable\r\n\r\n"
        b"Caf=C3=A9 au lait\r\n"
    )
    p = parse_message(raw)
    assert p.subject == "Café"
    assert "Café au lait" in p.body


def test_envelope_date_is_preferred_over_forged_date_header():
    raw = _plain(date="Mon, 01 Jan 2001 00:00:00 +0000")
    p = parse_message(raw, from_line="phisher@example.com Sat Nov 27 15:33:27 2004")
    assert p.date.year == 2004


def test_missing_or_absurd_date_gives_none():
    assert parse_message(_plain(date=None)).date is None
    assert parse_message(_plain(date="Thu, 01 Jan 1970 00:00:00 +0000")).date is None


def test_garbage_input_does_not_raise():
    p = parse_message(b"\x00\xff not an email at all")
    assert isinstance(p.body, str)


def test_html_to_text_handles_broken_markup():
    assert "hello" in html_to_text("<div><p>hello<b>world")
