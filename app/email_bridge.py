"""
app/email_bridge.py  —  Step 5: Gmail <-> RAG Bridge
=====================================================

AUTHENTICATION:
    Gmail uses standard IMAP/SMTP with an App Password.
    Do NOT use your real Google password here — generate an App Password:

    1. Go to https://myaccount.google.com/security
    2. Enable 2-Step Verification (required for App Passwords)
    3. Go to https://myaccount.google.com/apppasswords
    4. Create a password named "ArtemisBot"
    5. Copy the 16-character code into your .env

.env SETUP:
    GMAIL_ADDRESS=btechit211398@smvec.ac.in
    GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx   ← from step above (spaces OK)
    OPENAI_API_KEY=sk-...

HOW IT WORKS:
    Every 60 seconds:
      IMAP (Gmail)  →  search UNSEEN emails with subject "[ARTEMIS]"
                    →  extract body text as the RAG question
                    →  run through FAISS retrieve() + generate_answer()
                    →  SMTP (Gmail) reply back to the sender
                    →  mark original email as SEEN (no duplicate replies)

DEPENDENCIES (add to requirements.txt if not already present):
    openai
    faiss-cpu
    python-dotenv

QUICK TEST:
    Send an email to btechit211398@smvec.ac.in
    Subject: [ARTEMIS] Who are the Artemis II crew members?
    → You'll receive an AI reply within 60 seconds.
"""

from __future__ import annotations

import email
import imaplib
import json
import os
import re
import smtplib
import sys
import threading
import time
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# -- Add app/ to path so we can import the RAG pipeline ----------------------
APP_DIR  = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
sys.path.insert(0, str(APP_DIR))

# =============================================================================
#  CONFIGURATION
# =============================================================================

GMAIL_ADDRESS    = os.getenv("GMAIL_ADDRESS",    "")
GMAIL_APP_PW     = os.getenv("GMAIL_APP_PASSWORD", "").replace(" ", "")  # strip spaces

IMAP_HOST        = "imap.gmail.com"
IMAP_PORT        = 993   # SSL

SMTP_HOST        = "smtp.gmail.com"
SMTP_PORT        = 587   # STARTTLS

POLL_INTERVAL    = 60          # seconds between inbox checks
TRIGGER_PREFIX   = "[ARTEMIS]" # only emails with this in subject are processed
MAX_QUERY_CHARS  = 1000        # truncate very long emails

LOG_FILE         = ROOT_DIR / "email_bridge.log"


# =============================================================================
#  LOGGING
# =============================================================================

def log(msg: str) -> None:
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# =============================================================================
#  LOAD RAG PIPELINE
# =============================================================================

PIPELINE_READY = False
_run_rag       = None

try:
    from agents import run_agentic_pipeline  # type: ignore

    def _run_rag(question: str) -> str:
        """Run the full 3-agent pipeline (Agent 1 + Agent 2 + Agent 3)."""
        result = run_agentic_pipeline(
            query=question,
            save_report=True,
            check_external=True,
            save_web_to_corpus=False,   # don't pollute corpus from email queries
        )
        return result["final_answer"]

    PIPELINE_READY = True
    log("✅ 3-Agent pipeline loaded successfully (Agent 1 + Agent 2 + Agent 3).")

except Exception as exc:
    log(f"⚠️  3-Agent pipeline not ready: {exc}. Trying plain RAG fallback...")

    try:
        import faiss
        import numpy as np
        from openai import OpenAI
        from generator import generate_answer  # type: ignore

        DATA_DIR         = ROOT_DIR / "data"
        CHUNKS_PATH      = DATA_DIR / "chunks.json"
        FAISS_INDEX_PATH = DATA_DIR / "index.faiss"

        with open(CHUNKS_PATH, encoding="utf-8") as f:
            _chunks_list = json.load(f)

        _index  = faiss.read_index(str(FAISS_INDEX_PATH))
        _openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        def _retrieve(query: str, k: int = 5) -> list[dict]:
            resp = _openai.embeddings.create(
                model="text-embedding-3-small", input=query[:500]
            )
            vec = np.array(resp.data[0].embedding, dtype="float32").reshape(1, -1)
            faiss.normalize_L2(vec)
            scores, indices = _index.search(vec, k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if 0 <= idx < len(_chunks_list):
                    c = dict(_chunks_list[idx])
                    c["score"] = round(float(score), 4)
                    results.append(c)
            return results

        def _run_rag(question: str) -> str:  # type: ignore[misc]
            chunks = _retrieve(question)
            return generate_answer(question, chunks, history=None)

        PIPELINE_READY = True
        log("✅ Plain RAG fallback loaded.")

    except Exception as exc2:
        log(f"⚠️  Plain RAG also failed: {exc2}. Running in echo mode.")

        def _run_rag(question: str) -> str:  # type: ignore[misc]
            return (
                "The RAG pipeline is not currently loaded. "
                f"Your question was: '{question}'. "
                "Please contact the administrator."
            )


# =============================================================================
#  GMAIL IMAP  —  FETCH UNSEEN EMAILS
# =============================================================================

def _strip_html(html: str) -> str:
    """Very lightweight HTML → plain text (no extra dependencies)."""
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _decode_payload(msg: email.message.Message) -> str:
    """Extract plain-text body from a possibly multipart email."""
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            ctype   = part.get_content_type()
            charset = part.get_content_charset() or "utf-8"
            if ctype == "text/plain":
                body += part.get_payload(decode=True).decode(charset, errors="replace")
            elif ctype == "text/html" and not body:
                raw_html = part.get_payload(decode=True).decode(charset, errors="replace")
                body     = _strip_html(raw_html)
    else:
        charset = msg.get_content_charset() or "utf-8"
        payload = msg.get_payload(decode=True).decode(charset, errors="replace")
        ctype   = msg.get_content_type()
        body    = _strip_html(payload) if ctype == "text/html" else payload

    return body.strip()


def fetch_unseen_emails() -> list[dict]:
    """
    Connect to Gmail IMAP, search for UNSEEN messages that contain
    TRIGGER_PREFIX in the subject, return their metadata, and leave
    them as UNSEEN (we mark them after sending the reply).
    """
    results = []

    if not GMAIL_ADDRESS or not GMAIL_APP_PW:
        log("❌ GMAIL_ADDRESS or GMAIL_APP_PASSWORD not set in .env")
        return results

    try:
        with imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT) as imap:
            imap.login(GMAIL_ADDRESS, GMAIL_APP_PW)
            imap.select("INBOX")

            # Search for unseen emails containing trigger in subject
            _, data = imap.search(
                None,
                f'(UNSEEN SUBJECT "{TRIGGER_PREFIX}")'
            )
            message_ids = data[0].split()
            log(f"Found {len(message_ids)} new email(s) with prefix '{TRIGGER_PREFIX}'.")

            for mid in message_ids:
                _, msg_data = imap.fetch(mid, "(RFC822)")
                raw = msg_data[0][1]
                msg = email.message_from_bytes(raw)

                sender  = msg.get("From", "")
                subject = msg.get("Subject", "")
                body    = _decode_payload(msg)

                results.append({
                    "imap_id": mid,
                    "sender":  sender,
                    "subject": subject,
                    "body":    body[:MAX_QUERY_CHARS],
                })

    except imaplib.IMAP4.error as exc:
        log(f"❌ IMAP error: {exc}")
    except Exception as exc:
        log(f"❌ Unexpected IMAP error: {exc}")

    return results


def mark_as_read(imap_id: bytes) -> None:
    """Re-open the inbox and set the \\Seen flag on one message."""
    try:
        with imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT) as imap:
            imap.login(GMAIL_ADDRESS, GMAIL_APP_PW)
            imap.select("INBOX")
            imap.store(imap_id, "+FLAGS", "\\Seen")
    except Exception as exc:
        log(f"⚠️  Could not mark message as read: {exc}")


# =============================================================================
#  GMAIL SMTP  —  SEND REPLY
# =============================================================================

def send_reply(to_address: str, subject: str, answer: str) -> bool:
    """Send an HTML reply via Gmail SMTP (STARTTLS + App Password)."""
    if not GMAIL_ADDRESS or not GMAIL_APP_PW:
        log("❌ Cannot send reply: credentials not configured.")
        return False

    reply_subject = f"Re: {subject}" if not subject.startswith("Re:") else subject

    html_body = f"""
    <html><body style="font-family:Arial,sans-serif;color:#222;max-width:700px">
      <h2 style="color:#1a73e8">&#128640; Artemis II RAG Assistant</h2>
      <p>Here is the answer to your question:</p>
      <div style="background:#f0f7ff;border-left:4px solid #1a73e8;
                  padding:16px;border-radius:4px;white-space:pre-wrap">{answer}</div>
      <hr style="border:none;border-top:1px solid #ddd;margin-top:24px"/>
      <p style="font-size:12px;color:#888">
        Generated automatically by the Artemis II RAG Assistant.<br>
        Powered by: FAISS + GPT-4o-mini + Multi-Agent Pipeline<br>
        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
      </p>
    </body></html>
    """

    msg = MIMEMultipart("alternative")
    msg["From"]    = GMAIL_ADDRESS
    msg["To"]      = to_address
    msg["Subject"] = reply_subject
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.login(GMAIL_ADDRESS, GMAIL_APP_PW)
            smtp.sendmail(GMAIL_ADDRESS, to_address, msg.as_string())
        log(f"✅ Reply sent to {to_address}")
        return True
    except smtplib.SMTPAuthenticationError:
        log("❌ SMTP auth failed — check your App Password in .env")
        return False
    except Exception as exc:
        log(f"❌ Failed to send reply to {to_address}: {exc}")
        return False


# =============================================================================
#  QUERY HELPERS
# =============================================================================

def _clean_query(subject: str, body: str) -> str:
    """
    Remove the trigger prefix from subject and combine with body.
    Example:
        subject = "[ARTEMIS] Who are the crew members?"
        body    = "Please give details about each person."
        → "Who are the crew members?\n\nPlease give details about each person."
    """
    clean_subject = re.sub(
        re.escape(TRIGGER_PREFIX), "", subject, flags=re.IGNORECASE
    ).strip()

    if clean_subject and body:
        query = f"{clean_subject}\n\n{body}"
    elif clean_subject:
        query = clean_subject
    else:
        query = body

    return query.strip()[:MAX_QUERY_CHARS]


def _extract_address(sender_field: str) -> str:
    """Pull bare email address out of 'Name <addr>' or plain addr strings."""
    match = re.search(r"<(.+?)>", sender_field)
    return match.group(1) if match else sender_field.strip()


# =============================================================================
#  MAIN POLL LOOP
# =============================================================================

def poll_once() -> int:
    """Check inbox once. Returns number of emails processed."""
    emails = fetch_unseen_emails()

    for em in emails:
        sender  = em["sender"]
        subject = em["subject"]
        body    = em["body"]
        imap_id = em["imap_id"]

        log(f"Processing email from {sender} | Subject: {subject}")
        query = _clean_query(subject, body)
        log(f"   Query: '{query[:120]}'")

        if not query.strip():
            log("⚠️  Empty query — sending usage instructions.")
            answer = (
                "Your email was received but the body was empty.\n\n"
                "To ask a question, send an email with:\n"
                "  Subject: [ARTEMIS]\n"
                "  Body:    Your question here\n\n"
                "Example:\n"
                "  Subject: [ARTEMIS]\n"
                "  Body:    Who are the crew members of Artemis II?"
            )
            to_addr = _extract_address(sender)
            send_reply(to_address=to_addr, subject=subject, answer=answer)
            mark_as_read(imap_id)
            continue

        try:
            answer = _run_rag(query)
        except Exception as exc:
            answer = f"Sorry, an error occurred while processing your question: {exc}"
            log(f"❌ RAG error for {sender}: {exc}")

        to_addr = _extract_address(sender)
        send_reply(to_address=to_addr, subject=subject, answer=answer)
        mark_as_read(imap_id)

    return len(emails)


def run_poll_loop(stop_event: threading.Event | None = None) -> None:
    log(f"🚀 Email Bridge started. Polling every {POLL_INTERVAL}s.")
    log(f"   Inbox  : {GMAIL_ADDRESS}")
    log(f"   Trigger: emails with subject containing '{TRIGGER_PREFIX}'")
    log(f"   Auth   : Gmail App Password (IMAP SSL + SMTP STARTTLS)")
    log(f"   RAG    : {'READY ✅' if PIPELINE_READY else 'ECHO MODE ⚠️'}")
    log(f"   Log    : {LOG_FILE}")

    while True:
        if stop_event and stop_event.is_set():
            log("Stop event received. Shutting down.")
            break

        try:
            count = poll_once()
            if count == 0:
                log(f"💤 No new emails. Next check in {POLL_INTERVAL}s.")
        except Exception as exc:
            log(f"❌ Poll loop error: {exc}")

        # Sleep in 1-second ticks so stop_event is checked promptly
        for _ in range(POLL_INTERVAL):
            if stop_event and stop_event.is_set():
                break
            time.sleep(1)


def start_in_background() -> threading.Event:
    """Start the email bridge as a daemon thread alongside the Gradio UI."""
    stop_event = threading.Event()
    threading.Thread(
        target=run_poll_loop,
        args=(stop_event,),
        daemon=True,
        name="EmailBridge",
    ).start()
    log("Email bridge running as background thread.")
    return stop_event


# =============================================================================
#  ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("=" * 62)
    print("  Artemis II — Email Bridge (Gmail Edition)")
    print(f"  Inbox  : {GMAIL_ADDRESS or '(not set — add GMAIL_ADDRESS to .env)'}")
    print(f"  Trigger: emails with '{TRIGGER_PREFIX}' in subject")
    print(f"  Poll   : every {POLL_INTERVAL} seconds")
    print(f"  RAG    : {'READY ✅' if PIPELINE_READY else 'ECHO MODE ⚠️'}")
    print("=" * 62)

    if not GMAIL_ADDRESS or not GMAIL_APP_PW:
        print()
        print("  ⚠️  Missing credentials. Add these to your .env file:")
        print()
        print("    GMAIL_ADDRESS=your.address@gmail.com")
        print("    GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx")
        print()
        print("  How to get an App Password:")
        print("    1. https://myaccount.google.com/security")
        print("       → Enable 2-Step Verification")
        print("    2. https://myaccount.google.com/apppasswords")
        print("       → Create password named 'ArtemisBot'")
        print("       → Copy the 16-character code into .env")
        print()
    else:
        print()
        print("  How to test:")
        print(f"    Send email to: {GMAIL_ADDRESS}")
        print(f"    Subject: '{TRIGGER_PREFIX} Who are the Artemis II crew members?'")
        print()

    run_poll_loop()