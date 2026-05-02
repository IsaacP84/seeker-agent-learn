"""
monitor_training.py
Runs every 30 minutes while bin/app.exe is alive.
Sends an email with seeker.png, recent logs, and a brief training assessment.
"""

import json
import os
import re
import ssl
import smtplib
import subprocess
import time
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR    = r"C:\Github\Magic-Engine\install-release"
LOGS_DIR    = os.path.join(BASE_DIR, "logs")
SEEKER_LOG  = os.path.join(BASE_DIR, "bin", "runs", "seeker.log")
SEEKER_PNG  = os.path.join(BASE_DIR, "bin", "runs", "seeker.png")
SEEKER_PT   = os.path.join(BASE_DIR, "bin", "runs", "seeker.pt")
HYPERPARAMS = os.path.join(BASE_DIR, "assets", "hyperparameters.yml")

# ── Email config ─────────────────────────────────────────────────────────────
SMTP_SERVER      = "smtp.gmail.com"
SMTP_PORT        = 465

_creds_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "credentials.json")
with open(_creds_path, "r") as _f:
    _creds = json.load(_f)
SENDER_EMAIL     = _creds["sender_email"]
SENDER_PASSWORD  = _creds["sender_password"]
RECIPIENT_EMAIL  = _creds["recipient_email"]

# ── Timing ───────────────────────────────────────────────────────────────────
CHECK_INTERVAL_SEC = 10 * 60   # 30 minutes between emails
POLL_INTERVAL_SEC  = 60        # check app.exe every 60 s while waiting


# ── Helpers ──────────────────────────────────────────────────────────────────

def is_app_running() -> bool:
    result = subprocess.run(
        ["tasklist", "/FI", "IMAGENAME eq app.exe"],
        capture_output=True, text=True
    )
    return "app.exe" in result.stdout


def latest_log() -> str:
    """Return path of the most recently modified file in logs/."""
    try:
        files = [
            os.path.join(LOGS_DIR, f)
            for f in os.listdir(LOGS_DIR)
            if os.path.isfile(os.path.join(LOGS_DIR, f))
        ]
        return max(files, key=os.path.getmtime) if files else ""
    except Exception:
        return ""


def tail(path: str, n: int = 60) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[-n:])
    except Exception as e:
        return f"[Could not read {path}: {e}]"


def load_checkpoint_metrics() -> dict:
    """
    Load rewards_per_episode and epsilon from the .pt checkpoint.
    Returns empty dict on failure (torch may not be installed here).
    """
    try:
        import torch
        ckpt = torch.load(SEEKER_PT, map_location="cpu", weights_only=False)
        rewards = ckpt.get("rewards_per_episode", [])
        epsilon_history = ckpt.get("epsilon_history", [])
        epsilon = ckpt.get("epsilon", None)
        if epsilon is None and epsilon_history:
            epsilon = epsilon_history[-1]
        return {
            "episodes":       len(rewards),
            "rewards":        rewards,
            "epsilon":        epsilon,
        }
    except Exception:
        return {}


def parse_log_metrics(log_text: str) -> dict:
    """Fallback: parse episode/reward/epsilon from the text log."""
    episodes = re.findall(r"Episode\s+(\d+)", log_text)
    rewards  = re.findall(r"reward=\s*([-\d.]+)", log_text)
    epsilons = re.findall(r"epsilon=([\d.]+)", log_text)
    out = {}
    if episodes:
        out["episodes"] = int(episodes[-1])
    if rewards:
        out["rewards"] = [float(r) for r in rewards]
    if epsilons:
        out["epsilon"] = float(epsilons[-1])
    return out


def generate_assessment(metrics: dict, hyperparams_text: str) -> str:
    """Produce a <100-word training status summary."""
    episodes = metrics.get("episodes", 0)
    rewards  = metrics.get("rewards", [])
    epsilon  = metrics.get("epsilon")

    parts = []

    if episodes:
        parts.append(f"{episodes} episodes completed.")
    else:
        parts.append("Training just started — no episodes logged yet.")
        return " ".join(parts)

    if rewards:
        recent = rewards[-10:]
        avg    = sum(recent) / len(recent)
        best   = max(rewards)
        parts.append(f"Recent avg reward: {avg:.1f} (best: {best:.1f}).")

        # Simple trend: compare last 10 vs previous 10
        if len(rewards) >= 20:
            prev_avg = sum(rewards[-20:-10]) / 10
            delta    = avg - prev_avg
            if delta > 1:
                parts.append(f"Improving (+{delta:.1f} over last 10 ep).")
            elif delta < -1:
                parts.append(f"Declining ({delta:.1f} over last 10 ep).")
            else:
                parts.append("Stable trend.")

    if epsilon is not None:
        if epsilon > 0.5:
            phase = "still exploring heavily"
        elif epsilon > 0.15:
            phase = "transitioning to exploitation"
        else:
            phase = "near epsilon floor — exploiting learned policy"
        parts.append(f"Epsilon: {epsilon:.4f} ({phase}).")

    return " ".join(parts)


def send_email(subject: str, body: str, png_path: str | None = None) -> None:
    msg = MIMEMultipart()
    msg["From"]    = SENDER_EMAIL
    msg["To"]      = RECIPIENT_EMAIL
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    if png_path and os.path.exists(png_path):
        with open(png_path, "rb") as f:
            img_data = f.read()
        img = MIMEImage(img_data, name=os.path.basename(png_path))
        img.add_header(
            "Content-Disposition", "attachment",
            filename=os.path.basename(png_path)
        )
        msg.attach(img)

    ctx = ssl.create_default_context()
    with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=ctx) as server:
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Email sent: {subject}")


def build_and_send(stopped: bool = False) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Metrics — prefer checkpoint, fall back to log text
    metrics = load_checkpoint_metrics()
    seeker_log_text = tail(SEEKER_LOG, 80)
    if not metrics:
        metrics = parse_log_metrics(seeker_log_text)

    hyperparams_text = tail(HYPERPARAMS, 999)
    assessment = generate_assessment(metrics, hyperparams_text)

    if stopped:
        subject = f"Training STOPPED [{timestamp}]"
        assessment = "app.exe has stopped. Final snapshot below.\n\n" + assessment
    else:
        subject = f"Training Update [{timestamp}]"

    body = (
        f"=== ASSESSMENT ===\n"
        f"{assessment}\n\n"
        f"=== SEEKER LOG (recent) ===\n"
        f"{seeker_log_text}\n"
        f"=== APP LOG ({os.path.basename(latest_log()) or 'none'}) ===\n"
        f"{tail(latest_log(), 60)}"
    )

    send_email(subject, body, SEEKER_PNG)


# ── Main loop ────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Monitor started. "
          f"Sending every {CHECK_INTERVAL_SEC // 60} min while app.exe is running.")

    if not is_app_running():
        print("app.exe is not running. Exiting immediately.")
        return

    while True:
        if not is_app_running():
            print("app.exe stopped. Sending final notification.")
            build_and_send(stopped=True)
            break

        build_and_send()

        # Wait CHECK_INTERVAL_SEC, polling for app death every POLL_INTERVAL_SEC
        elapsed = 0
        while elapsed < CHECK_INTERVAL_SEC:
            time.sleep(POLL_INTERVAL_SEC)
            elapsed += POLL_INTERVAL_SEC
            if not is_app_running():
                print("app.exe stopped during wait. Sending final notification.")
                build_and_send(stopped=True)
                return

    print("Monitor exiting.")


if __name__ == "__main__":
    main()
