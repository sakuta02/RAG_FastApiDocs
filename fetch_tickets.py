"""Собираем закрытые issues FastAPI с ответами — получается корпус типа тикетов поддержки."""

import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

REPO = "fastapi/fastapi"
MAX_ISSUES = 40
OUTPUT_FILE = Path("tickets/github_issues.jsonl")


def main():
    headers = build_headers()
    print("Тащим закрытые issues...")
    issues = fetch_closed_issues(headers)
    print(f"Нашли {len(issues)} закрытых issues с ответами, качаем комментарии...")
    tickets = [to_ticket(issue, fetch_comments(issue["comments_url"], headers)) for issue in issues]
    print(f"Сохраняем {len(tickets)} тикетов...")
    save_tickets(tickets)
    print("Готово.")


def build_headers():
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    else:
        print("GITHUB_TOKEN не задан - лимит 60 запросов в час, увы.")
    return headers


def fetch_closed_issues(headers):
    issues = []
    page = 1
    while len(issues) < MAX_ISSUES:
        resp = requests.get(
            f"https://api.github.com/repos/{REPO}/issues",
            params={"state": "closed", "per_page": 100, "page": page},
            headers=headers,
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        # issues-эндпоинт отдаёт ещё и pull request'ы, их отсеиваем
        issues.extend(i for i in batch if "pull_request" not in i and i["comments"] > 0)
        page += 1
    return issues[:MAX_ISSUES]


def fetch_comments(comments_url, headers):
    resp = requests.get(comments_url, headers=headers)
    resp.raise_for_status()
    return [c["body"] for c in resp.json()]


def to_ticket(issue, comments):
    return {
        "number": issue["number"],
        "title": issue["title"],
        "body": issue["body"] or "",
        "comments": comments,
        "url": issue["html_url"],
        "labels": [label["name"] for label in issue["labels"]],
    }


def save_tickets(tickets):
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        for ticket in tickets:
            f.write(json.dumps(ticket) + "\n")


if __name__ == "__main__":
    main()
