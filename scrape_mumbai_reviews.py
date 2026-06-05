"""
scrape_mumbai_reviews.py
Scrapes real Google Maps reviews for all 163+ Mumbai petrol pump stations.

Strategy
--------
1. For each station (name + lat/lng from mumbai_petrol_pumps.xlsx):
   - Navigate to Google Maps search URL with coordinates
   - Accept consent if shown, set language to English
   - Click the first result (the petrol pump)
   - Click the "Reviews" tab
   - Scroll the reviews panel to load all reviews
   - Extract: author, star rating, text, relative date
2. Save each station's reviews to CSV incrementally so progress survives restarts.
3. Merge all into mumbai_petrol_reviews.csv at the end.

Usage
-----
    python scrape_mumbai_reviews.py                 # scrape all stations
    python scrape_mumbai_reviews.py --limit 10      # scrape first 10 stations only (test)
    python scrape_mumbai_reviews.py --resume        # skip already-scraped stations
    python scrape_mumbai_reviews.py --merge-only    # just re-merge existing per-station files
"""

import argparse
import csv
import hashlib
import json
import os
import re
import time
import random
import logging
import traceback
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException, NoSuchElementException, StaleElementReferenceException,
    ElementClickInterceptedException, InvalidSessionIdException,
    WebDriverException,
)

# ── Configuration ──────────────────────────────────────────────────────────────

PUMPS_FILE   = "mumbai_petrol_pumps.xlsx"
OUT_DIR      = "scraped_reviews"          # per-station JSON cache
MERGED_CSV   = "mumbai_petrol_reviews.csv"

MAX_REVIEWS_PER_STATION = 300            # high cap — Google lazily loads ~10 per scroll batch
SCROLL_PAUSE             = 1.5           # seconds between scroll steps
MAX_SCROLL_ATTEMPTS      = 60            # 60 scrolls × ~10 reviews = up to ~600 reviews
NO_NEW_REVIEWS_LIMIT     = 5            # stop scrolling after this many empty scroll rounds
PAGE_LOAD_TIMEOUT        = 20            # seconds
ELEMENT_WAIT             = 12            # seconds for element waits
INTER_STATION_SLEEP      = (2, 4)        # random sleep range between stations

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ── Chrome driver setup ────────────────────────────────────────────────────────

def make_driver(headless=True):
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    opts.add_argument("--window-size=1280,900")
    opts.add_argument(
        "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
    opts.add_argument("--lang=en-US,en;q=0.9")
    opts.add_argument("--disable-extensions")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--log-level=3")
    driver = webdriver.Chrome(options=opts)
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"},
    )
    driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
    return driver


# ── Google Maps helpers ────────────────────────────────────────────────────────

def handle_consent(driver):
    """Dismiss cookie / consent dialog if present."""
    for selector in [
        "button[aria-label*='Accept']",
        "button[aria-label*='Agree']",
        "form[action*='consent'] button",
        "#L2AGLb",               # Google's 'I agree' button id
    ]:
        try:
            btn = driver.find_element(By.CSS_SELECTOR, selector)
            btn.click()
            time.sleep(1)
            return True
        except Exception:
            pass
    return False


def safe_get(driver, url, timeout=PAGE_LOAD_TIMEOUT):
    """Navigate to url, stop the page after timeout regardless of exception type."""
    try:
        driver.set_page_load_timeout(timeout)
        driver.get(url)
    except Exception:
        pass
    # Always stop any pending load so nothing hangs downstream
    try:
        driver.execute_script("window.stop();")
    except Exception:
        pass


def search_station(driver, name, lat, lng):
    """
    Navigate to Google Maps and land on the station's place page.
    Returns True if we successfully landed on a place page.
    """
    query = quote(f"{name} petrol pump")
    url = (
        f"https://www.google.com/maps/search/{query}"
        f"/@{lat},{lng},17z?hl=en"
    )
    safe_get(driver, url)
    time.sleep(2)
    handle_consent(driver)

    # If on a search results page, click the first result
    current = driver.current_url
    if "/search/" in current or "search?" in current:
        try:
            result_links = driver.find_elements(By.CSS_SELECTOR, "a[href*='/maps/place/']")
            if result_links:
                driver.execute_script("arguments[0].click();", result_links[0])
                time.sleep(2.5)
            else:
                return False
        except Exception:
            return False

    return "/maps/place/" in driver.current_url or "ludocid" in driver.current_url


def click_reviews_tab(driver):
    """Find and click the Reviews tab on a place page. Returns True on success."""
    wait = WebDriverWait(driver, ELEMENT_WAIT)

    # Try multiple selectors for the Reviews button/tab
    selectors = [
        "button[aria-label*='Reviews']",
        "button[data-tab-index='1']",
        "//button[contains(@aria-label,'Reviews')]",
        "//button[normalize-space()='Reviews']",
    ]
    for sel in selectors:
        try:
            if sel.startswith("//"):
                el = driver.find_element(By.XPATH, sel)
            else:
                el = driver.find_element(By.CSS_SELECTOR, sel)
            driver.execute_script("arguments[0].click();", el)
            time.sleep(2)
            return True
        except Exception:
            pass

    # Fallback: look for tab buttons containing 'review' text
    try:
        tabs = driver.find_elements(By.CSS_SELECTOR, "button[role='tab'], div[role='tab']")
        for tab in tabs:
            if "review" in tab.text.lower() or "review" in tab.get_attribute("aria-label", "").lower():
                driver.execute_script("arguments[0].click();", tab)
                time.sleep(2)
                return True
    except Exception:
        pass

    return False


def get_reviews_scrollable_panel(driver):
    """Return the scrollable reviews container element."""
    selectors = [
        "div[role='main'] div[tabindex='-1']",
        "div.m6QErb.DxyBCb",
        "div.m6QErb",
        "div[aria-label*='Reviews']",
    ]
    for sel in selectors:
        try:
            els = driver.find_elements(By.CSS_SELECTOR, sel)
            for el in els:
                if el.size.get("height", 0) > 100:
                    return el
        except Exception:
            pass
    return None


def count_review_blocks(driver):
    """Count currently loaded review blocks on the page."""
    for sel in ["div[data-review-id]", "div.jftiEf"]:
        try:
            els = driver.find_elements(By.CSS_SELECTOR, sel)
            if els:
                return len(els)
        except Exception:
            pass
    return 0


def sort_reviews_by_newest(driver):
    """Click 'Sort reviews' button → select Newest."""
    try:
        btn = driver.find_element(By.CSS_SELECTOR, "button[aria-label='Sort reviews']")
        driver.execute_script("arguments[0].click();", btn)
        time.sleep(1.5)
    except Exception:
        return False

    # Dropdown appears — look for the Newest option
    for opt_sel in [
        "//div[@role='menuitemradio'][contains(.,'Newest')]",
        "//li[@role='menuitemradio'][contains(.,'Newest')]",
        "//div[@role='option'][contains(.,'Newest')]",
        "//li[contains(.,'Newest')]",
        "div[data-index='1'][role='menuitemradio']",
    ]:
        try:
            opt = (driver.find_element(By.XPATH, opt_sel)
                   if opt_sel.startswith("//")
                   else driver.find_element(By.CSS_SELECTOR, opt_sel))
            driver.execute_script("arguments[0].click();", opt)
            time.sleep(2.5)
            return True
        except Exception:
            pass
    return False


def scroll_reviews(driver, max_attempts=MAX_SCROLL_ATTEMPTS):
    """
    Scroll the reviews panel, stopping when no new reviews appear
    after NO_NEW_REVIEWS_LIMIT consecutive scroll rounds.
    """
    panel = get_reviews_scrollable_panel(driver)

    prev_count = 0
    no_new_count = 0

    for _ in range(max_attempts):
        if panel:
            try:
                driver.execute_script("arguments[0].scrollTop += 1200;", panel)
            except Exception:
                driver.execute_script("window.scrollBy(0, 1200);")
        else:
            driver.execute_script("window.scrollBy(0, 1200);")

        time.sleep(SCROLL_PAUSE)

        new_count = count_review_blocks(driver)
        if new_count == prev_count:
            no_new_count += 1
            if no_new_count >= NO_NEW_REVIEWS_LIMIT:
                break
        else:
            no_new_count = 0
        prev_count = new_count

        # Re-acquire panel in case DOM was refreshed
        if panel:
            try:
                panel = get_reviews_scrollable_panel(driver)
            except Exception:
                panel = None


def expand_review_texts(driver):
    """Click all 'More' buttons to expand truncated review texts."""
    for sel in ["button[aria-label='See more']", "button.w8nwRe", ".review-more-link"]:
        try:
            btns = driver.find_elements(By.CSS_SELECTOR, sel)
            for btn in btns:
                try:
                    driver.execute_script("arguments[0].click();", btn)
                    time.sleep(0.3)
                except Exception:
                    pass
        except Exception:
            pass


def parse_star_from_aria(aria_label):
    """Extract numeric star rating from aria-label like '4 stars'."""
    if not aria_label:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*star", aria_label, re.IGNORECASE)
    if m:
        return float(m.group(1))
    # fallback: first number
    m2 = re.search(r"(\d+(?:\.\d+)?)", aria_label)
    return float(m2.group(1)) if m2 else None


def extract_reviews_from_page(driver, station_name, station_address,
                               station_rating, station_review_count,
                               brand, category, zone, lat, lng):
    """
    Parse review elements from the current page.
    Returns list of dicts.
    """
    reviews = []
    collected_at = datetime.utcnow().isoformat() + "+00:00"

    # All known review block selectors
    review_block_selectors = [
        "div[data-review-id]",
        "div.jftiEf",
        "div.review-block",
        "div[class*='review']",
    ]

    blocks = []
    for sel in review_block_selectors:
        try:
            found = driver.find_elements(By.CSS_SELECTOR, sel)
            if found:
                blocks = found
                break
        except Exception:
            pass

    if not blocks:
        return []

    for block in blocks[:MAX_REVIEWS_PER_STATION]:
        try:
            # ── Author ──
            author = ""
            for sel in ["button[data-review-id]", ".d4r55", "span[class*='author']",
                        ".section-review-owner-name", "button.al6Kxe"]:
                try:
                    el = block.find_element(By.CSS_SELECTOR, sel)
                    author = el.text.strip()
                    if author:
                        break
                except Exception:
                    pass

            # ── Star rating ──
            stars = None
            for sel in ["span[aria-label*='star']", "span[role='img'][aria-label*='star']",
                        ".kvMYJc", "span[class*='star']"]:
                try:
                    el = block.find_element(By.CSS_SELECTOR, sel)
                    stars = parse_star_from_aria(el.get_attribute("aria-label") or "")
                    if stars is not None:
                        break
                except Exception:
                    pass

            # ── Review text ──
            text = ""
            for sel in ["span.wiI7pd", ".review-full-text", "span[class*='review']",
                        ".section-review-text", "div[data-review-id] span"]:
                try:
                    el = block.find_element(By.CSS_SELECTOR, sel)
                    text = el.text.strip()
                    if text:
                        break
                except Exception:
                    pass

            # ── Date ──
            date_relative = ""
            for sel in ["span.rsqaWe", ".section-review-publish-date", "span[class*='date']"]:
                try:
                    el = block.find_element(By.CSS_SELECTOR, sel)
                    date_relative = el.text.strip()
                    if date_relative:
                        break
                except Exception:
                    pass

            if not (author or text):
                continue

            review_id = hashlib.md5(
                f"{station_name}|{lat}|{lng}|{author}|{text}".encode()
            ).hexdigest()

            reviews.append({
                "review_id":           review_id,
                "station_name":        station_name,
                "station_address":     station_address,
                "station_rating":      station_rating,
                "station_reviews":     station_review_count,
                "brand":               brand,
                "category":            category,
                "zone":                zone,
                "lat":                 lat,
                "lng":                 lng,
                "author_name":         author,
                "rating":              stars,
                "text":                text,
                "date_relative":       date_relative,
                "date_iso":            collected_at,
                "collected_at":        collected_at,
                "source":              "google_maps_scrape",
            })

        except (StaleElementReferenceException, Exception):
            continue

    return reviews


# ── Brand / Zone helpers ───────────────────────────────────────────────────────

BRAND_MAP = {
    "hindustan petroleum": "HP", "hp ": "HP", "hpcl": "HP",
    "bharat petroleum": "BPCL", "bpcl": "BPCL",
    "indianoil": "IndianOil", "indian oil": "IndianOil", "iocl": "IndianOil",
    "mahanagar gas": "Mahanagar Gas", "mgl": "Mahanagar Gas",
    "essar": "Essar", "shell": "Shell", "nayara": "Nayara",
    "reliance": "Reliance",
}


def extract_brand(title):
    t = str(title).lower()
    for key, val in BRAND_MAP.items():
        if key in t:
            return val
    return "Other"


def assign_zone(lat, lng):
    try:
        lat, lng = float(lat), float(lng)
    except Exception:
        return "Mumbai"
    if lat < 18.970:
        return "South Mumbai"
    elif lat < 19.050:
        return "Central Mumbai"
    elif lat < 19.150:
        return "Western Suburbs" if lng < 72.880 else "Eastern Suburbs"
    else:
        return "North Suburbs"


# ── Main scraping loop ─────────────────────────────────────────────────────────

def already_scraped(out_dir, station_key):
    path = Path(out_dir) / f"{station_key}.json"
    return path.exists()


def save_station(out_dir, station_key, reviews):
    path = Path(out_dir) / f"{station_key}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(reviews, f, ensure_ascii=False, indent=2)


def load_station(out_dir, station_key):
    path = Path(out_dir) / f"{station_key}.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def station_key(title, idx):
    return f"{idx:03d}_{re.sub(r'[^a-z0-9]', '_', title.lower())[:40]}"


def merge_all(out_dir, merged_csv):
    """Merge all per-station JSON files into one CSV."""
    all_reviews = []
    for path in sorted(Path(out_dir).glob("*.json")):
        with open(path, encoding="utf-8") as f:
            reviews = json.load(f)
        all_reviews.extend(reviews)
    if not all_reviews:
        log.warning("No reviews found to merge.")
        return pd.DataFrame()
    df = pd.DataFrame(all_reviews).drop_duplicates("review_id")
    df.to_csv(merged_csv, index=False)
    log.info(f"Merged {len(df):,} reviews from {len(list(Path(out_dir).glob('*.json')))} stations → {merged_csv}")
    return df




def _navigate_to_station(drv, title, address, lat, lng):
    """Land on the station's Maps place page. Returns True on success."""
    landed = search_station(drv, title, lat, lng)
    if not landed:
        log.warning("  -> Fallback: searching by address")
        query2 = quote(f"{title} {address[:60]}")
        safe_get(drv, f"https://www.google.com/maps/search/{query2}?hl=en")
        time.sleep(2)
        handle_consent(drv)
        try:
            res = drv.find_elements(By.CSS_SELECTOR, "a[href*='/maps/place/']")
            if res:
                drv.execute_script("arguments[0].click();", res[0])
                time.sleep(2.5)
        except Exception:
            pass
    return "/maps/place/" in drv.current_url or "ludocid" in drv.current_url


def _collect_reviews(drv, title, address, rating, rc, brand, cat, zone, lat, lng):
    """Scroll fully, expand all, extract reviews from current view."""
    expand_review_texts(drv)
    scroll_reviews(drv)
    expand_review_texts(drv)  # expand any newly loaded ones
    return extract_reviews_from_page(drv, title, address, rating, rc, brand, cat, zone, lat, lng)


def _do_scrape_one(drv, title, address, lat, lng, rating, rc, brand, cat, zone):
    """
    Scrape a station twice — first with default (Most Relevant) sort,
    then switch to Newest sort — and merge unique reviews by review_id.
    This typically doubles the review yield per station.
    """
    _navigate_to_station(drv, title, address, lat, lng)
    place_url = drv.current_url   # remember so we can reload for second sort

    # ── Pass 1: Most Relevant (default) ───────────────────────────────────────
    click_reviews_tab(drv)
    time.sleep(2)
    reviews_relevant = _collect_reviews(drv, title, address, rating, rc, brand, cat, zone, lat, lng)
    log.info(f"    [Relevant sort] {len(reviews_relevant)} reviews")

    # ── Pass 2: Newest ────────────────────────────────────────────────────────
    # Don't reload — scroll the reviews panel back to top, then sort
    reviews_newest = []
    try:
        panel = get_reviews_scrollable_panel(drv)
        if panel:
            drv.execute_script("arguments[0].scrollTop = 0;", panel)
        else:
            drv.execute_script("window.scrollTo(0,0);")
        time.sleep(1)
        sorted_ok = sort_reviews_by_newest(drv)
        if sorted_ok:
            reviews_newest = _collect_reviews(drv, title, address, rating, rc, brand, cat, zone, lat, lng)
            log.info(f"    [Newest sort]   {len(reviews_newest)} reviews")
        else:
            log.info(f"    [Newest sort]   sort button not found, skipping second pass")
    except Exception as e:
        log.warning(f"    [Newest sort]   failed ({e}), keeping first-pass results only")

    # ── Merge unique reviews ──────────────────────────────────────────────────
    seen = {r["review_id"] for r in reviews_relevant}
    combined = list(reviews_relevant)
    for r in reviews_newest:
        if r["review_id"] not in seen:
            combined.append(r)
            seen.add(r["review_id"])

    return combined


def scrape_all(limit=None, resume=True, headless=True, merge_only=False):
    Path(OUT_DIR).mkdir(exist_ok=True)

    if merge_only:
        return merge_all(OUT_DIR, MERGED_CSV)

    pumps = pd.read_excel(PUMPS_FILE)
    log.info(f"Loaded {len(pumps)} stations from {PUMPS_FILE}")
    if limit:
        pumps = pumps.head(limit)

    PROACTIVE_RESTART_EVERY = 15   # restart Chrome every N stations to avoid OOM crashes

    rows = list(pumps.iterrows())
    total = len(rows)
    scraped_count = 0
    failed = []
    driver = make_driver(headless=headless)
    i = 0

    while i < total:
        idx, row = rows[i]
        title   = str(row.get("title", f"Station_{idx}")).strip()
        address = str(row.get("address", "")).strip().replace("\n", " ").replace("\ue0c8", "").strip()
        lat     = float(row.get("lat", row.get("latitude_clean", 19.076)))
        lng     = float(row.get("lng", row.get("longitude_clean", 72.877)))
        rating  = row.get("totalScore", None)
        rc      = row.get("reviewsCount", 0)
        brand   = extract_brand(title)
        zone    = assign_zone(lat, lng)
        cat     = "CNG Station" if "compressed" in str(row.get("categoryName", "")).lower() else "Petrol Pump"
        key     = station_key(title, idx)

        if resume and already_scraped(OUT_DIR, key):
            existing = load_station(OUT_DIR, key)
            log.info(f"[{i+1}/{total}] SKIP (cached {len(existing)} reviews): {title}")
            i += 1
            continue

        # Proactive Chrome restart every N stations to prevent OOM crashes
        if scraped_count > 0 and scraped_count % PROACTIVE_RESTART_EVERY == 0:
            log.info(f"  [Proactive restart after {scraped_count} stations to free memory]")
            try:
                driver.quit()
            except Exception:
                pass
            driver = make_driver(headless=headless)

        log.info(f"[{i+1}/{total}] Scraping: {title}")
        reviews = []

        try:
            reviews = _do_scrape_one(driver, title, address, lat, lng, rating, rc, brand, cat, zone)
            log.info(f"  -> {len(reviews)} reviews collected")

        except (InvalidSessionIdException, WebDriverException) as e:
            log.warning(f"  -> Driver crashed ({type(e).__name__}), restarting Chrome...")
            try:
                driver.quit()
            except Exception:
                pass
            driver = make_driver(headless=headless)
            try:
                reviews = _do_scrape_one(driver, title, address, lat, lng, rating, rc, brand, cat, zone)
                log.info(f"  -> Retry success: {len(reviews)} reviews")
            except Exception as e2:
                log.error(f"  -> Retry also failed: {e2}")
                failed.append(title)
                reviews = []

        except Exception as e:
            log.error(f"  -> ERROR: {e}")
            failed.append(title)
            reviews = []

        save_station(OUT_DIR, key, reviews)
        scraped_count += 1
        i += 1
        time.sleep(random.uniform(*INTER_STATION_SLEEP))

    try:
        driver.quit()
    except Exception:
        pass

    log.info(f"\nScraping complete. Stations attempted: {scraped_count}. Failed: {len(failed)}")
    if failed:
        log.warning("Failed stations:\n" + "\n".join(f"  * {s}" for s in failed))

    return merge_all(OUT_DIR, MERGED_CSV)



# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Google Maps reviews for Mumbai petrol pumps")
    parser.add_argument("--limit",      type=int, default=None, help="Max stations to scrape (test mode)")
    parser.add_argument("--no-resume",  action="store_true",   help="Re-scrape even if cached")
    parser.add_argument("--visible",    action="store_true",   help="Run Chrome in visible mode (non-headless)")
    parser.add_argument("--merge-only", action="store_true",   help="Skip scraping; just merge existing files")
    args = parser.parse_args()

    df = scrape_all(
        limit      = args.limit,
        resume     = not args.no_resume,
        headless   = not args.visible,
        merge_only = args.merge_only,
    )
    print(f"\nFinal dataset: {len(df):,} reviews in {MERGED_CSV}")
