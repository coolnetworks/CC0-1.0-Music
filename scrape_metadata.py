#!/usr/bin/env python3
"""
Web metadata scraper for CC0-1.0 Music Collection.

Scrapes genre, instrumental/vocal, mood, and tag metadata from the original
source websites for ~6,292 MP3 files. Supports resume via progress file.

Sources:
  - freemusicarchive.org (5,181 files) - genre, instrumental, AI-generated
  - chosic.com (798 files) - tags (genre + mood)
  - freepd.com (302 files) - category (via Wayback Machine)
  - pixbay.com (11 files) - tags (via Wayback Machine)
"""

import json
import os
import re
import time
import random
import difflib
import logging
import urllib.parse
from pathlib import Path
from datetime import date

import requests
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).parent
PROGRESS_FILE = BASE_DIR / "scrape_progress.json"
OUTPUT_FILE = BASE_DIR / "web_metadata.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(BASE_DIR / "scrape.log"),
    ],
)
log = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"
)
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})
_last_request_time = time.monotonic()


# ---------------------------------------------------------------------------
# Progress / Resume helpers
# ---------------------------------------------------------------------------

def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {
        "freepd": {"done": False, "results": []},
        "fma_search": {"done": False, "artist_tracks": {}, "completed_artists": []},
        "fma_scrape": {"done": False, "results": [], "completed_urls": []},
        "chosic": {"done": False, "results": [], "completed_ids": []},
        "pixabay": {"done": False, "results": []},
    }


def save_progress(progress):
    tmp = str(PROGRESS_FILE) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(progress, f)
    os.replace(tmp, PROGRESS_FILE)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def fetch(url, retries=3, backoff=2.0, timeout=30):
    """Fetch URL with rate limiting, retries, and exponential backoff."""
    global _last_request_time
    for attempt in range(retries):
        elapsed = time.monotonic() - _last_request_time
        min_gap = 1.5 + random.uniform(0, 0.5)
        sleep_time = max(0, min_gap - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)
        _last_request_time = time.monotonic()
        try:
            resp = SESSION.get(url, timeout=timeout)
            if resp.status_code == 200:
                return resp
            if resp.status_code in (429, 403, 503):
                wait = backoff * (2 ** attempt) + random.uniform(0, 1)
                log.warning("HTTP %d on %s, retrying in %.1fs", resp.status_code, url, wait)
                time.sleep(wait)
                continue
            if resp.status_code == 404:
                log.debug("404: %s", url)
                return None
            log.warning("HTTP %d: %s", resp.status_code, url)
            return None
        except requests.RequestException as e:
            wait = backoff * (2 ** attempt)
            log.warning("Request error on %s: %s, retrying in %.1fs", url, e, wait)
            time.sleep(wait)
    log.error("Failed after %d retries: %s", retries, url)
    return None


def soup(resp):
    """Parse response into BeautifulSoup."""
    return BeautifulSoup(resp.text, "html.parser")


# ---------------------------------------------------------------------------
# File listing helpers
# ---------------------------------------------------------------------------

def list_files(source_dir):
    """List all .mp3 files in a source directory."""
    d = BASE_DIR / source_dir
    if not d.exists():
        return []
    return sorted(f.name for f in d.iterdir() if f.suffix.lower() == ".mp3")


# ---------------------------------------------------------------------------
# 1. FreePD via Wayback Machine
# ---------------------------------------------------------------------------

FREEPD_CATEGORIES = [
    "comedy", "electronic", "epic", "horror",
    "misc", "romantic", "scoring", "upbeat", "world",
]


def scrape_freepd(progress):
    if progress["freepd"]["done"]:
        log.info("FreePD: already done, skipping")
        return progress["freepd"]["results"]

    log.info("=== FreePD via Wayback Machine ===")
    local_files = list_files("freepd.com")
    local_lower = {f.lower(): f for f in local_files}
    log.info("FreePD: %d local files", len(local_files))

    results = []
    matched_files = set()

    for category in FREEPD_CATEGORIES:
        url = f"https://web.archive.org/web/2024/https://freepd.com/{category}.php"
        log.info("Fetching FreePD category: %s", category)
        resp = fetch(url)
        if not resp:
            log.warning("Failed to fetch FreePD category: %s", category)
            continue

        page = soup(resp)

        # Extract MP3 filenames from source tags and download links
        filenames = set()

        # Method 1: <source> tags
        for source_tag in page.find_all("source", type="audio/mpeg"):
            src = source_tag.get("src", "")
            m = re.search(r"/music/(.+\.mp3)", urllib.parse.unquote(src))
            if m:
                filenames.add(m.group(1))

        # Method 2: download button hrefs
        for a in page.find_all("a", class_="downloadButton"):
            href = a.get("href", "")
            m = re.search(r"/music/(.+\.mp3)", urllib.parse.unquote(href))
            if m:
                filenames.add(m.group(1))

        # Method 3: any href with .mp3
        for a in page.find_all("a", href=True):
            href = a["href"]
            m = re.search(r"/music/(.+\.mp3)", urllib.parse.unquote(href))
            if m:
                filenames.add(m.group(1))

        log.info("  %s: found %d tracks", category, len(filenames))

        for fname in filenames:
            # Try exact match, then case-insensitive
            if fname in local_files:
                match_name = fname
            elif fname.lower() in local_lower:
                match_name = local_lower[fname.lower()]
            else:
                log.debug("  No local match for: %s", fname)
                continue

            if match_name in matched_files:
                continue
            matched_files.add(match_name)

            results.append({
                "filename": match_name,
                "source": "freepd.com",
                "source_url": f"https://freepd.com/{category}.php",
                "genres": [],
                "instrumental": None,
                "ai_generated": None,
                "tags": [category],
                "category": category,
                "match_confidence": 1.0,
            })

    # Add unmatched local files with empty metadata
    for f in local_files:
        if f not in matched_files:
            results.append({
                "filename": f,
                "source": "freepd.com",
                "source_url": None,
                "genres": [],
                "instrumental": None,
                "ai_generated": None,
                "tags": [],
                "category": None,
                "match_confidence": 0.0,
            })

    log.info("FreePD: %d matched, %d unmatched",
             len(matched_files), len(local_files) - len(matched_files))

    progress["freepd"]["done"] = True
    progress["freepd"]["results"] = results
    save_progress(progress)
    return results


# ---------------------------------------------------------------------------
# 2. FMA - Phase A: Discover track URLs via artist search
# ---------------------------------------------------------------------------

def parse_fma_filename(filename):
    """Parse 'Artist Name - Track Title.mp3' into (artist, title)."""
    name = filename.rsplit(".", 1)[0]  # strip .mp3
    if " - " in name:
        parts = name.split(" - ", 1)
        return parts[0].strip(), parts[1].strip()
    return None, name.strip()


def normalize_for_match(s):
    """Normalize string for fuzzy matching: lowercase, strip punctuation."""
    s = s.lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def fma_search_artist(artist_name):
    """Search FMA for an artist, return list of (track_title_slug, track_url, genres)."""
    encoded = urllib.parse.quote(artist_name)
    tracks = []
    page_num = 1

    while True:
        url = f"https://freemusicarchive.org/search?adv=1&quicksearch={encoded}&page={page_num}"
        resp = fetch(url)
        if not resp:
            break

        page = soup(resp)
        items = page.find_all("div", class_="play-item")
        if not items:
            break

        for item in items:
            # Extract track URL
            track_link = item.find("span", class_="ptxt-track")
            if not track_link:
                continue
            a = track_link.find("a")
            if not a:
                continue
            track_url = a.get("href", "")
            if not track_url.startswith("http"):
                track_url = "https://freemusicarchive.org" + track_url
            track_title = a.get_text(strip=True)

            # Check artist matches
            artist_span = item.find("span", class_="ptxt-artist")
            if artist_span:
                artist_link = artist_span.find("a")
                if artist_link:
                    page_artist = artist_link.get_text(strip=True)
                    # Only keep results where artist matches
                    ratio = difflib.SequenceMatcher(
                        None,
                        normalize_for_match(artist_name),
                        normalize_for_match(page_artist),
                    ).ratio()
                    if ratio < 0.7:
                        continue

            # Extract genres from search results (only place genres appear)
            genres = []
            genre_span = item.find("span", class_="ptxt-genre")
            if genre_span:
                for ga in genre_span.find_all("a"):
                    g = ga.get_text(strip=True)
                    if g:
                        genres.append(g)

            tracks.append({
                "title": track_title,
                "url": track_url,
                "genres": genres,
            })

        # Check for next page
        pagination = page.find("div", class_=re.compile(r"pagination"))
        if not pagination:
            break
        # Look for a "next" link or link with page_num+1
        next_link = None
        for a in pagination.find_all("a"):
            href = a.get("href", "")
            if f"page={page_num + 1}" in href:
                next_link = href
                break
        if not next_link:
            break
        page_num += 1

    return tracks


def scrape_fma_search(progress):
    """Phase A: Search FMA for all artists, collect track URLs and genres."""
    if progress["fma_search"]["done"]:
        log.info("FMA Search: already done, skipping")
        return progress["fma_search"]["artist_tracks"]

    log.info("=== FMA Phase A: Artist Search ===")
    local_files = list_files("freemusicarchive.org")

    # Extract unique artists
    artists = set()
    for f in local_files:
        artist, _ = parse_fma_filename(f)
        if artist:
            artists.add(artist)

    artists = sorted(artists)
    completed = set(progress["fma_search"]["completed_artists"])
    log.info("FMA: %d unique artists, %d already searched", len(artists), len(completed))

    artist_tracks = progress["fma_search"]["artist_tracks"]
    save_counter = 0

    for i, artist in enumerate(artists):
        if artist in completed:
            continue

        log.info("  [%d/%d] Searching: %s", i + 1, len(artists), artist)
        tracks = fma_search_artist(artist)
        if tracks:
            artist_tracks[artist] = tracks
            log.info("    Found %d tracks", len(tracks))
        else:
            log.info("    No results")

        completed.add(artist)
        save_counter += 1

        if save_counter >= 20:
            progress["fma_search"]["completed_artists"] = list(completed)
            progress["fma_search"]["artist_tracks"] = artist_tracks
            save_progress(progress)
            save_counter = 0

    progress["fma_search"]["done"] = True
    progress["fma_search"]["completed_artists"] = list(completed)
    progress["fma_search"]["artist_tracks"] = artist_tracks
    save_progress(progress)
    return artist_tracks


# ---------------------------------------------------------------------------
# 2. FMA - Phase B: Scrape individual track pages
# ---------------------------------------------------------------------------

def scrape_fma_track_page(url):
    """Scrape a single FMA track page for instrumental and AI-generated flags."""
    resp = fetch(url)
    if not resp:
        return {}

    page = soup(resp)
    info = {}

    # Look for metadata in grid layout divs
    grids = page.find_all("div", class_=re.compile(r"grid"))
    for grid in grids:
        spans = grid.find_all("span")
        if len(spans) >= 2:
            label = spans[0].get_text(strip=True).lower()
            value = spans[1].get_text(strip=True)

            if "instrumental" in label:
                info["instrumental"] = value.lower() in ("yes", "true", "1")
            elif "ai generated" in label or "ai-generated" in label:
                info["ai_generated"] = value.lower() in ("yes", "true", "1")

    return info


def match_fma_files(local_files, artist_tracks):
    """Match local filenames to discovered FMA track URLs."""
    matches = {}  # filename -> {url, genres, title, match_confidence}

    for filename in local_files:
        artist, title = parse_fma_filename(filename)
        if not artist or artist not in artist_tracks:
            continue

        tracks = artist_tracks[artist]
        best_match = None
        best_score = 0

        norm_title = normalize_for_match(title)

        for track in tracks:
            # Compare against track title from search results
            norm_track = normalize_for_match(track["title"])
            score = difflib.SequenceMatcher(None, norm_title, norm_track).ratio()

            # Also compare against URL slug
            url_slug = track["url"].rstrip("/").split("/")[-1]
            url_slug = urllib.parse.unquote(url_slug).replace("-", " ")
            norm_slug = normalize_for_match(url_slug)
            slug_score = difflib.SequenceMatcher(None, norm_title, norm_slug).ratio()

            score = max(score, slug_score)
            if score > best_score:
                best_score = score
                best_match = track

        if best_match and best_score >= 0.7:
            matches[filename] = {
                "url": best_match["url"],
                "genres": best_match["genres"],
                "title": best_match["title"],
                "match_confidence": round(best_score, 3),
            }

    return matches


def scrape_fma_tracks(progress, artist_tracks):
    """Phase B: Scrape individual track pages for instrumental/AI flags."""
    if progress["fma_scrape"]["done"]:
        log.info("FMA Track Scrape: already done, skipping")
        return progress["fma_scrape"]["results"]

    log.info("=== FMA Phase B: Track Page Scraping ===")
    local_files = list_files("freemusicarchive.org")

    # Build matches
    matches = match_fma_files(local_files, artist_tracks)
    log.info("FMA: %d/%d files matched to track URLs", len(matches), len(local_files))

    completed_urls = set(progress["fma_scrape"]["completed_urls"])
    results = progress["fma_scrape"]["results"]
    results_by_file = {r["filename"]: r for r in results}
    save_counter = 0
    total = len(matches)

    for i, (filename, match) in enumerate(matches.items()):
        if filename in results_by_file:
            continue

        url = match["url"]
        log.info("  [%d/%d] Scraping: %s", i + 1, total, filename[:60])

        track_info = {}
        if url not in completed_urls:
            track_info = scrape_fma_track_page(url)
            completed_urls.add(url)
        else:
            # Reuse info from a previously scraped result with the same URL
            for r in results:
                if r.get("source_url") == url:
                    track_info = {
                        "instrumental": r.get("instrumental"),
                        "ai_generated": r.get("ai_generated"),
                    }
                    break

        result = {
            "filename": filename,
            "source": "freemusicarchive.org",
            "source_url": match["url"],
            "genres": match["genres"],
            "instrumental": track_info.get("instrumental"),
            "ai_generated": track_info.get("ai_generated"),
            "tags": [],
            "match_confidence": match["match_confidence"],
        }
        results.append(result)
        results_by_file[filename] = result
        save_counter += 1

        if save_counter >= 50:
            progress["fma_scrape"]["results"] = results
            progress["fma_scrape"]["completed_urls"] = list(completed_urls)
            save_progress(progress)
            save_counter = 0

    # Add unmatched FMA files
    for f in local_files:
        if f not in results_by_file:
            results.append({
                "filename": f,
                "source": "freemusicarchive.org",
                "source_url": None,
                "genres": [],
                "instrumental": None,
                "ai_generated": None,
                "tags": [],
                "match_confidence": 0.0,
            })

    progress["fma_scrape"]["done"] = True
    progress["fma_scrape"]["results"] = results
    progress["fma_scrape"]["completed_urls"] = list(completed_urls)
    save_progress(progress)

    matched = sum(1 for r in results if r["match_confidence"] > 0)
    log.info("FMA: %d matched, %d unmatched", matched, len(results) - matched)
    return results


# ---------------------------------------------------------------------------
# 3. Chosic
# ---------------------------------------------------------------------------

def parse_chosic_filename(filename):
    """Parse 'Artist_-_Track_Title(chosic.com).mp3' into (artist, title)."""
    name = filename.rsplit(".", 1)[0]  # strip .mp3
    name = re.sub(r"\(chosic\.com\)$", "", name).strip()
    name = name.replace("_", " ")
    if " - " in name:
        # Handle numbered tracks: "Artist - 01 - Title"
        parts = name.split(" - ", 1)
        artist = parts[0].strip()
        rest = parts[1].strip()
        # Strip leading track number
        rest = re.sub(r"^\d+\s*-\s*", "", rest).strip()
        return artist, rest
    return None, name.strip()


def scrape_chosic_listing_page(page_num):
    """Scrape a single Chosic listing page, return list of track dicts."""
    url = f"https://www.chosic.com/free-music/all/?page={page_num}"
    resp = fetch(url)
    if not resp:
        return []

    page = soup(resp)

    # Collect track and artist links in DOM order: track, artist, track, artist...
    elements = []
    for a in page.find_all("a"):
        href = a.get("href", "")
        text = a.get_text(strip=True)
        m = re.search(r"/download-audio/(\d+)", href)
        if m and text and text.lower() not in ("download", "free download"):
            elements.append(("track", m.group(1), text))
        elif "keyword=" in href and "&artist" in href:
            elements.append(("artist", None, text))

    # Pair tracks with their following artist
    tracks = []
    i = 0
    while i < len(elements):
        if elements[i][0] == "track":
            track_id = elements[i][1]
            title = elements[i][2]
            artist = None
            if i + 1 < len(elements) and elements[i + 1][0] == "artist":
                artist = elements[i + 1][2]
                i += 1
            tracks.append({"id": track_id, "title": title, "artist": artist})
        i += 1

    # Deduplicate by ID
    seen = set()
    deduped = []
    for t in tracks:
        if t["id"] not in seen:
            seen.add(t["id"])
            deduped.append(t)

    return deduped


def scrape_chosic_track(track_id):
    """Scrape a Chosic track detail page for tags."""
    url = f"https://www.chosic.com/download-audio/{track_id}/"
    resp = fetch(url)
    if not resp:
        return []

    page = soup(resp)
    tags = []

    # Noise tags that aren't useful genre/mood info
    noise_tags = {"sound effects", "ringtones", "medical"}

    # Tags appear as links to /free-music/{tag-slug}/
    for a in page.find_all("a", href=re.compile(r"/free-music/[^/]+/$")):
        href = a.get("href", "")
        # Exclude the main listing links
        if "/free-music/all/" in href:
            continue
        tag = a.get_text(strip=True)
        if tag and len(tag) > 1 and tag.lower() not in noise_tags:
            tags.append(tag)

    # Deduplicate while preserving order
    seen = set()
    result = []
    for t in tags:
        tl = t.lower()
        if tl not in seen:
            seen.add(tl)
            result.append(t)

    return result


def scrape_chosic(progress):
    """Scrape Chosic for track tags."""
    if progress["chosic"]["done"]:
        log.info("Chosic: already done, skipping")
        return progress["chosic"]["results"]

    log.info("=== Chosic ===")
    local_files = list_files("chosic.com")
    log.info("Chosic: %d local files", len(local_files))

    # Parse local filenames
    local_parsed = {}
    for f in local_files:
        artist, title = parse_chosic_filename(f)
        local_parsed[f] = (artist, title)

    # Phase 1: Discover track IDs from listing pages
    log.info("Chosic: Discovering track IDs from listing pages...")
    all_listing_tracks = []
    page_num = 1
    empty_pages = 0

    while empty_pages < 3:  # stop after 3 consecutive empty pages
        log.info("  Fetching listing page %d...", page_num)
        tracks = scrape_chosic_listing_page(page_num)
        if not tracks:
            empty_pages += 1
        else:
            empty_pages = 0
            all_listing_tracks.extend(tracks)
        page_num += 1

    log.info("Chosic: Found %d unique tracks from listing pages",
             len({t["id"] for t in all_listing_tracks}))

    # Phase 2: Match local files to listing tracks and scrape tags
    completed_ids = set(progress["chosic"]["completed_ids"])
    results = progress["chosic"]["results"]
    results_by_file = {r["filename"]: r for r in results}
    save_counter = 0

    for i, filename in enumerate(local_files):
        if filename in results_by_file:
            continue

        artist, title = local_parsed[filename]
        norm_title = normalize_for_match(title)
        norm_artist = normalize_for_match(artist) if artist else ""

        # Try to find matching track using artist+title
        best_match = None
        best_score = 0

        for track in all_listing_tracks:
            norm_track_title = normalize_for_match(track["title"])
            title_score = difflib.SequenceMatcher(
                None, norm_title, norm_track_title
            ).ratio()

            # Boost score if artist also matches
            if norm_artist and track.get("artist"):
                norm_track_artist = normalize_for_match(track["artist"])
                artist_score = difflib.SequenceMatcher(
                    None, norm_artist, norm_track_artist
                ).ratio()
                # Combined: 60% title, 40% artist
                score = 0.6 * title_score + 0.4 * artist_score
            else:
                score = title_score

            if score > best_score:
                best_score = score
                best_match = track

        tags = []
        source_url = None
        track_id = None

        if best_match and best_score >= 0.6:
            track_id = best_match["id"]
            source_url = f"https://www.chosic.com/download-audio/{track_id}/"

            if track_id not in completed_ids:
                log.info("  [%d/%d] Scraping tags: %s (id=%s, score=%.2f)",
                         i + 1, len(local_files), filename[:50], track_id, best_score)
                tags = scrape_chosic_track(track_id)
                completed_ids.add(track_id)
            else:
                # Already scraped this track ID, find the existing result with same ID
                for r in results:
                    if r.get("source_url") == source_url:
                        tags = r.get("tags", [])
                        break
        else:
            log.debug("  No listing match for: %s (best=%.2f)", filename[:50], best_score)

        result = {
            "filename": filename,
            "source": "chosic.com",
            "source_url": source_url,
            "genres": [],
            "instrumental": None,
            "ai_generated": None,
            "tags": tags,
            "match_confidence": round(best_score, 3) if best_match else 0.0,
        }
        results.append(result)
        results_by_file[filename] = result
        save_counter += 1

        if save_counter >= 50:
            progress["chosic"]["results"] = results
            progress["chosic"]["completed_ids"] = list(completed_ids)
            save_progress(progress)
            save_counter = 0

    progress["chosic"]["done"] = True
    progress["chosic"]["results"] = results
    progress["chosic"]["completed_ids"] = list(completed_ids)
    save_progress(progress)

    matched = sum(1 for r in results if r["match_confidence"] > 0)
    log.info("Chosic: %d matched, %d unmatched", matched, len(results) - matched)
    return results


# ---------------------------------------------------------------------------
# 4. Pixabay via Wayback Machine
# ---------------------------------------------------------------------------

def parse_pixabay_filename(filename):
    """Parse 'slug-with-dashes-12345.mp3' into (slug, id)."""
    name = filename.rsplit(".", 1)[0]
    m = re.match(r"^(.+)-(\d+)$", name)
    if m:
        return m.group(1), m.group(2)
    return name, None


def scrape_pixabay(progress):
    """Scrape Pixabay metadata via Wayback Machine."""
    if progress["pixabay"]["done"]:
        log.info("Pixabay: already done, skipping")
        return progress["pixabay"]["results"]

    log.info("=== Pixabay via Wayback Machine ===")
    local_files = list_files("pixbay.com")
    log.info("Pixabay: %d local files", len(local_files))

    results = []

    for filename in local_files:
        slug, pid = parse_pixabay_filename(filename)
        if not pid:
            log.warning("  Can't parse ID from: %s", filename)
            results.append({
                "filename": filename,
                "source": "pixbay.com",
                "source_url": None,
                "genres": [],
                "instrumental": None,
                "ai_generated": None,
                "tags": [],
                "match_confidence": 0.0,
            })
            continue

        # Try Wayback Machine
        original_url = f"https://pixabay.com/music/{slug}-{pid}/"
        wayback_url = f"https://web.archive.org/web/2024/{original_url}"
        log.info("  Fetching: %s", filename)
        resp = fetch(wayback_url)

        tags = []
        if resp:
            page = soup(resp)
            # Look for tag elements - pixabay uses various patterns
            for a in page.find_all("a", href=re.compile(r"/music/search/")):
                tag = a.get_text(strip=True)
                if tag and len(tag) > 1:
                    tags.append(tag)

            # Also check meta keywords
            meta = page.find("meta", attrs={"name": "keywords"})
            if meta and meta.get("content"):
                for kw in meta["content"].split(","):
                    kw = kw.strip()
                    if kw and kw.lower() not in {t.lower() for t in tags}:
                        tags.append(kw)

        results.append({
            "filename": filename,
            "source": "pixbay.com",
            "source_url": original_url if resp else None,
            "genres": [],
            "instrumental": None,
            "ai_generated": None,
            "tags": tags,
            "match_confidence": 1.0 if resp else 0.0,
        })

    progress["pixabay"]["done"] = True
    progress["pixabay"]["results"] = results
    save_progress(progress)

    matched = sum(1 for r in results if r["match_confidence"] > 0)
    log.info("Pixabay: %d matched, %d unmatched", matched, len(results) - matched)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_output(freepd_results, fma_results, chosic_results, pixabay_results):
    """Combine all results into final output."""
    all_results = freepd_results + fma_results + chosic_results + pixabay_results

    total = len(all_results)
    matched = sum(1 for r in all_results if r["match_confidence"] > 0)

    output = {
        "scrape_date": str(date.today()),
        "stats": {
            "total": total,
            "matched": matched,
            "unmatched": total - matched,
            "by_source": {},
        },
        "results": all_results,
    }

    # Per-source stats
    sources = {}
    for r in all_results:
        src = r["source"]
        if src not in sources:
            sources[src] = {"total": 0, "matched": 0}
        sources[src]["total"] += 1
        if r["match_confidence"] > 0:
            sources[src]["matched"] += 1
    output["stats"]["by_source"] = sources

    return output


def main():
    log.info("Starting metadata scraper")
    progress = load_progress()

    # 1. FreePD (fastest - ~9 requests)
    freepd_results = scrape_freepd(progress)

    # 2. FMA Phase A: Search (slow - ~668 artists)
    artist_tracks = scrape_fma_search(progress)

    # 3. FMA Phase B: Scrape track pages (slowest - ~5181 pages)
    fma_results = scrape_fma_tracks(progress, artist_tracks)

    # 4. Chosic
    chosic_results = scrape_chosic(progress)

    # 5. Pixabay (quick - 11 files)
    pixabay_results = scrape_pixabay(progress)

    # Build and save final output
    output = build_output(freepd_results, fma_results, chosic_results, pixabay_results)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    log.info("=== Done ===")
    log.info("Total: %d, Matched: %d, Unmatched: %d",
             output["stats"]["total"],
             output["stats"]["matched"],
             output["stats"]["unmatched"])
    for src, stats in output["stats"]["by_source"].items():
        log.info("  %s: %d/%d matched", src, stats["matched"], stats["total"])
    log.info("Results saved to: %s", OUTPUT_FILE)


if __name__ == "__main__":
    main()
