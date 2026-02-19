from pathlib import Path

import requests
from playwright.sync_api import sync_playwright


def main() -> None:
    ui_url = "http://127.0.0.1:8501"
    api_url = "http://127.0.0.1:8000"

    vote_file = Path("votes.json")
    if vote_file.exists():
        vote_file.unlink()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(ui_url, wait_until="networkidle", timeout=60000)

        inp = page.get_by_placeholder("Ask demo question here...")
        inp.fill("recommend breathable lightweight running upper")
        inp.press("Enter")

        page.get_by_text("AI Summary", exact=False).wait_for(timeout=90000)
        page.get_by_role("button", name="Approve").first.click(timeout=20000)
        page.wait_for_timeout(1800)
        browser.close()

    data = requests.get(f"{api_url}/votes", timeout=10).json()
    print(f"votes_total={data['counts']['total']}")
    print(f"approved={data['counts']['approved']}")
    print(f"disapproved={data['counts']['disapproved']}")
    sample = next(iter(data["votes"].items())) if data["votes"] else None
    print(f"sample={sample}")


if __name__ == "__main__":
    main()
