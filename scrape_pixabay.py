#!/usr/bin/env python3
"""Scrape Pixabay music metadata using Playwright (headless Firefox)."""

import json
import re
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

BASE_DIR = Path(__file__).parent
PIXABAY_DIR = BASE_DIR / "pixbay.com"
OUTPUT_FILE = BASE_DIR / "web_metadata.json"

FILES = sorted(f.name for f in PIXABAY_DIR.iterdir() if f.suffix.lower() == ".mp3")


def parse_filename(filename):
    name = filename.rsplit(".", 1)[0]
    m = re.match(r"^(.+)-(\d+)$", name)
    if m:
        return m.group(1), m.group(2)
    return name, None


def scrape():
    results = []
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (X11; Linux x86_64; rv:134.0) Gecko/20100101 Firefox/134.0",
            locale="en-US",
        )
        page = context.new_page()

        for filename in FILES:
            slug, pid = parse_filename(filename)
            if not pid:
                print(f"  SKIP (no ID): {filename}")
                results.append({"filename": filename, "source": "pixbay.com", "tags": [], "match_confidence": 0.0})
                continue

            url = f"https://pixabay.com/music/{slug}-{pid}/"
            print(f"  Fetching: {filename}")
            print(f"    URL: {url}")

            tags = []
            source_url = None
            confidence = 0.0

            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
                time.sleep(2)

                # Extract tags from links matching /music/search/
                tag_els = page.query_selector_all('a[href*="/music/search/"]')
                for el in tag_els:
                    tag = el.inner_text().strip()
                    if tag and len(tag) > 1:
                        tags.append(tag)

                # Also try meta keywords
                meta = page.query_selector('meta[name="keywords"]')
                if meta:
                    content = meta.get_attribute("content") or ""
                    existing = {t.lower() for t in tags}
                    for kw in content.split(","):
                        kw = kw.strip()
                        if kw and kw.lower() not in existing:
                            tags.append(kw)

                if tags:
                    confidence = 1.0
                    source_url = url
                    print(f"    Tags: {tags}")
                else:
                    # Page loaded but no tags found - check if we got the right page
                    title = page.title()
                    print(f"    No tags found (title: {title})")
                    if "pixabay" in title.lower():
                        confidence = 0.5
                        source_url = url

            except Exception as e:
                print(f"    Error: {e}")

            results.append({
                "filename": filename,
                "source": "pixbay.com",
                "source_url": source_url,
                "genres": [],
                "instrumental": None,
                "ai_generated": None,
                "tags": tags,
                "match_confidence": confidence,
            })

            time.sleep(1)

        browser.close()

    return results


def update_output(pixabay_results):
    """Patch pixabay entries in existing web_metadata.json."""
    if not OUTPUT_FILE.exists():
        print("web_metadata.json not found, can't patch")
        return

    with open(OUTPUT_FILE) as f:
        data = json.load(f)

    # Build lookup by filename
    new_by_file = {r["filename"]: r for r in pixabay_results}

    updated = 0
    for entry in data["results"]:
        if entry["source"] == "pixbay.com" and entry["filename"] in new_by_file:
            new = new_by_file[entry["filename"]]
            entry["tags"] = new["tags"]
            entry["source_url"] = new["source_url"]
            entry["match_confidence"] = new["match_confidence"]
            updated += 1

    # Update stats
    matched = sum(1 for r in data["results"] if r["source"] == "pixbay.com" and r["match_confidence"] > 0)
    data["stats"]["by_source"]["pixbay.com"]["matched"] = matched
    data["stats"]["matched"] = sum(1 for r in data["results"] if r["match_confidence"] > 0)
    data["stats"]["unmatched"] = data["stats"]["total"] - data["stats"]["matched"]

    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nUpdated {updated} entries in web_metadata.json")
    print(f"Pixabay: {matched}/11 matched")


if __name__ == "__main__":
    print(f"Scraping {len(FILES)} Pixabay files...")
    results = scrape()
    update_output(results)
    print("Done!")
