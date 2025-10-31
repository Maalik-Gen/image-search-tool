import os
import subprocess
import smtplib
from email.mime.text import MIMEText

# ========== CONFIG ==========
REPO_PATH = r"C:\path\to\your\repo"  # <-- Change this
LAST_HASH_FILE = "last_commit.txt"
BRANCH = "main"

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_USER = "your_email@gmail.com"      # your sender email
EMAIL_PASS = "your_app_password"         # app password, not your login password
EMAIL_TO = ["maalik.hemani@genetech.co"]  # recipient(s)

# ============================

def get_latest_commit_hash(repo_path, branch):
    """Get the latest commit hash from the local git repo."""
    result = subprocess.run(
        ["git", "-C", repo_path, "rev-parse", branch],
        capture_output=True, text=True
    )
    return result.stdout.strip()

def send_email(subject, body):
    """Send an email notification."""
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_USER
    msg["To"] = ", ".join(EMAIL_TO)

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.send_message(msg)

def main():
    latest_hash = get_latest_commit_hash(REPO_PATH, BRANCH)

    if not os.path.exists(LAST_HASH_FILE):
        with open(LAST_HASH_FILE, "w") as f:
            f.write(latest_hash)
        print("Initialized commit tracker.")
        return

    with open(LAST_HASH_FILE, "r") as f:
        last_hash = f.read().strip()

    if latest_hash != last_hash:
        print("New commit detected!")

        # Get commit details
        commit_message = subprocess.run(
            ["git", "-C", REPO_PATH, "log", "-1", "--pretty=%B"],
            capture_output=True, text=True
        ).stdout.strip()

        author_name = subprocess.run(
            ["git", "-C", REPO_PATH, "log", "-1", "--pretty=%an"],
            capture_output=True, text=True
        ).stdout.strip()

        body = f"""New commit pushed to {BRANCH} branch.

Commit: {latest_hash}
Author: {author_name}
Message: {commit_message}
Repository: {REPO_PATH}
"""
        send_email("New Commit Alert", body)

        with open(LAST_HASH_FILE, "w") as f:
            f.write(latest_hash)
    else:
        print("No new commits detected.")

if __name__ == "__main__":
    main()
