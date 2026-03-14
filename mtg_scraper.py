#!/usr/bin/env python3
"""
Unified MTG Deck Scraper
Supports AetherHub (Standard BO1) and MTGGoldfish (Standard) as sources.

Usage:
  python3 mtg_scraper.py --source all          # scrape both (default)
  python3 mtg_scraper.py --source aetherhub    # AetherHub only
  python3 mtg_scraper.py --source mtggoldfish  # MTGGoldfish only

  python3 mtg_scraper.py --source all --meta 75 --pages 6 --max-decks 80
  python3 mtg_scraper.py --backfill-only
  python3 mtg_scraper.py --backfill-only --fix-colors
"""

import os
import re
import json
import sqlite3
import time
import uuid
import random
import logging
from collections import Counter
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps

import requests
from bs4 import BeautifulSoup

# ── Config ────────────────────────────────────────────────────────────────────
FORGE_DECK_DIR = Path('/home/toaster/.forge/decks/constructed')
DB_PATH        = Path('./decks_cache.db')

AETHERHUB_META_URL = "https://aetherhub.com/Metagame/Standard-BO1/"
AETHERHUB_USER_URL = "https://aetherhub.com/Decks/Standard-BO1/"

GOLDFISH_META_URL  = "https://www.mtggoldfish.com/metagame/standard/full#paper"
GOLDFISH_BASE_URL  = "https://www.mtggoldfish.com"

BASIC_LANDS = {'Plains', 'Island', 'Swamp', 'Mountain', 'Forest', 'Wastes'}

# Slug prefixes keep sources distinct in the DB even if deck names overlap
AETHERHUB_PREFIX = ""          # existing slugs have no prefix — keep unchanged
GOLDFISH_PREFIX  = "goldfish-"

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def retry(max_attempts=3, delay=1, backoff=2, exceptions=(Exception,)):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt, current_delay = 1, delay
            while attempt <= max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        raise
                    log.warning(f"Retry {attempt}/{max_attempts} for {func.__name__}: {e}")
                    time.sleep(current_delay + random.uniform(0, current_delay * 0.1))
                    current_delay *= backoff
                    attempt += 1
        return wrapper
    return decorator


def parse_maindeck(text: str, max_cards: int = 60) -> List[str]:
    """
    Parse a deck text block into a flat list of card names (repeated by qty).
    Stops at sideboard marker. Caps at max_cards.
    """
    sideboard_pos = len(text)
    for marker in ['Sideboard', 'SIDEBOARD', 'Side Board', '// Sideboard', 'sb:']:
        pos = text.lower().find(marker.lower())
        if pos != -1 and pos < sideboard_pos:
            sideboard_pos = pos

    lines = text[:sideboard_pos].strip().split('\n')
    card_names: List[str] = []

    _NOISE = {
        'summon','cast','draw','play','add','tap','the','a','an','attack','block',
        'exile','discard','sacrifice','return','create','counter','destroy','target',
        'choose','reveal','search','shuffle','deep','loyal',
    }
    _ALLOWLIST = {'Opt', 'Niv', 'Ion', 'Bow'}

    for line in lines:
        line = line.strip()
        if not line:
            continue
        m = re.match(r'^(\d+)x?\s+(.+)$', line)
        if not m:
            continue
        qty       = int(m.group(1))
        card_name = m.group(2).strip()
        card_name = re.sub(r'\s*[\(\[][A-Z0-9]{2,6}[\)\]]\s*\d*\s*$', '', card_name).strip()

        if not (1 <= qty <= 99):
            continue
        if re.match(r'^\d+$', card_name) or re.match(r'^[A-Z]{2,6}$', card_name):
            continue
        words = card_name.split()
        if len(words) == 1 and card_name not in _ALLOWLIST:
            if len(card_name) < 4 or card_name.lower() in _NOISE:
                continue
        if len(card_name) < 3:
            continue

        remaining = max_cards - len(card_names)
        if remaining <= 0:
            break
        card_names.extend([card_name] * min(qty, remaining))

    return card_names


def colors_from_name(name: str) -> str:
    """Best-effort color extraction from a deck name string."""
    n = name.lower()
    color_map = {
        'mono white':'W', 'mono blue':'U', 'mono black':'B',
        'mono red':'R',   'mono green':'G',
        'azorius':'WU',   'dimir':'UB',    'rakdos':'BR',
        'gruul':'RG',     'selesnya':'WG', 'orzhov':'WB',
        'izzet':'UR',     'golgari':'BG',  'boros':'WR',
        'simic':'UG',     'esper':'WUB',   'grixis':'UBR',
        'jund':'BRG',     'naya':'WRG',    'bant':'WUG',
        'mardu':'WBR',    'temur':'URG',   'abzan':'WBG',
        'jeskai':'WUR',   'sultai':'UBG',  'four color':'WUBR',
        'five color':'WUBRG',
    }
    for kw, colors in color_map.items():
        if kw in n:
            return colors
    found = ''
    for letter, word in [('W','white'),('U','blue'),('B','black'),('R','red'),('G','green')]:
        if word in n and letter not in found:
            found += letter
    return found


# ═════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Deck:
    slug:         str
    url:          str
    name:         str
    colors:       str
    is_meta:      bool
    tier:         Optional[int]
    file_hash:    str
    card_names:   List[str]
    tags:         List[str]
    last_updated: str


# ═════════════════════════════════════════════════════════════════════════════
# DATABASE
# ═════════════════════════════════════════════════════════════════════════════

class DeckDatabase:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._connect() as conn:
            c = conn.cursor()
            c.execute('''
                CREATE TABLE IF NOT EXISTS decks (
                    slug TEXT PRIMARY KEY, url TEXT UNIQUE, name TEXT,
                    colors TEXT, is_meta BOOLEAN, tier INTEGER,
                    file_hash TEXT, card_names TEXT, tags TEXT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            c.execute('''
                CREATE TABLE IF NOT EXISTS cards (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL, mana_cost TEXT, cmc INTEGER,
                    colors TEXT, type_line TEXT, types TEXT, subtypes TEXT,
                    oracle_text TEXT, keywords TEXT, rarity TEXT,
                    is_basic_land BOOLEAN DEFAULT 0, user_quantity INTEGER DEFAULT 1,
                    set_code TEXT, collector_number TEXT, scryfall_id TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            c.execute("CREATE INDEX IF NOT EXISTS idx_cards_name   ON cards(name)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_cards_colors ON cards(colors)")

    def exists(self, slug: str) -> bool:
        with self._connect() as conn:
            return conn.execute(
                "SELECT 1 FROM decks WHERE slug = ?", (slug,)
            ).fetchone() is not None

    def add_deck(self, deck: Deck):
        with self._connect() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO decks
                (slug,url,name,colors,is_meta,tier,file_hash,card_names,tags,last_updated)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            ''', (
                deck.slug, deck.url, deck.name, deck.colors, deck.is_meta,
                deck.tier, deck.file_hash, json.dumps(deck.card_names),
                json.dumps(deck.tags), deck.last_updated,
            ))

    def get_all_decks(self) -> List[Deck]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM decks").fetchall()
        return [
            Deck(slug=r[0], url=r[1], name=r[2], colors=r[3], is_meta=bool(r[4]),
                 tier=r[5], file_hash=r[6], card_names=json.loads(r[7]),
                 tags=json.loads(r[8]), last_updated=r[9])
            for r in rows
        ]

    def get_count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM decks").fetchone()[0]

    def get_count_by_source(self) -> Dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT tags, COUNT(*) FROM decks GROUP BY tags"
            ).fetchall()
        counts: Dict[str, int] = {'aetherhub': 0, 'goldfish': 0, 'other': 0}
        for tags_json, n in rows:
            try:
                tags = json.loads(tags_json) if tags_json else []
            except Exception:
                tags = []
            if 'goldfish' in tags:
                counts['goldfish'] += n
            elif 'aetherhub' in tags:
                counts['aetherhub'] += n
            else:
                counts['other'] += n
        return counts

    def get_all_card_names(self) -> Set[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT card_names FROM decks").fetchall()
        all_cards: Set[str] = set()
        for (blob,) in rows:
            if blob:
                try:
                    all_cards.update(json.loads(blob))
                except Exception:
                    pass
        return all_cards

    def get_missing_cards(self, card_names: List[str]) -> List[str]:
        if not card_names:
            return []
        with self._connect() as conn:
            ph = ','.join('?' * len(card_names))
            rows = conn.execute(
                f"SELECT name FROM cards WHERE name IN ({ph})", card_names
            ).fetchall()
        existing = {r[0] for r in rows}
        return [n for n in card_names if n not in existing]

    def card_exists(self, name: str) -> bool:
        with self._connect() as conn:
            return conn.execute(
                "SELECT 1 FROM cards WHERE name = ?", (name,)
            ).fetchone() is not None

    def reset_empty_colors(self) -> int:
        with self._connect() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM cards WHERE colors = '[]' OR colors IS NULL"
            ).fetchone()[0]
            conn.execute("DELETE FROM cards WHERE colors = '[]' OR colors IS NULL")
        return count

    def insert_card_data(self, name: str, data: dict):
        colors = data.get('colors', [])
        if not colors and 'card_faces' in data:
            colors = data['card_faces'][0].get('colors', [])
        type_line = data.get('type_line', '')
        types, subtypes = type_line, ''
        if '—' in type_line:
            types, subtypes = [s.strip() for s in type_line.split('—', 1)]
        with self._connect() as conn:
            try:
                conn.execute('''
                    INSERT OR REPLACE INTO cards
                    (name,mana_cost,cmc,colors,type_line,types,subtypes,oracle_text,
                     keywords,rarity,is_basic_land,user_quantity,set_code,collector_number,scryfall_id)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ''', (
                    name, data.get('mana_cost',''), data.get('cmc',0),
                    json.dumps(colors), type_line, types, subtypes,
                    data.get('oracle_text',''), json.dumps(data.get('keywords',[])),
                    data.get('rarity',''), 'Basic' in type_line and 'Land' in type_line,
                    1, data.get('set',''), data.get('collector_number',''), data.get('id',''),
                ))
            except sqlite3.Error as e:
                log.error(f"Error inserting card '{name}': {e}")


# ═════════════════════════════════════════════════════════════════════════════
# SCRYFALL ENRICHER
# ═════════════════════════════════════════════════════════════════════════════

class ScryfallEnricher:
    _REQUEST_DELAY = 0.12

    def __init__(self):
        self._cache: Dict[str, dict] = {}

    def _extract_colors(self, data: dict) -> List[str]:
        colors = data.get('colors', [])
        if not colors and 'card_faces' in data:
            colors = data['card_faces'][0].get('colors', [])
        if not colors:
            colors = list(set(re.findall(r'\{([WUBRG])\}', data.get('mana_cost',''))))
        if not colors:
            colors = data.get('color_identity', [])
        return colors

    @retry(max_attempts=3, delay=0.5, exceptions=(requests.RequestException,))
    def fetch_card(self, card_name: str) -> Optional[dict]:
        if card_name in self._cache:
            return self._cache[card_name]
        resp = requests.get(
            "https://api.scryfall.com/cards/named",
            params={"exact": card_name}, timeout=10,
        )
        if resp.status_code == 429:
            wait = int(resp.headers.get('Retry-After', 5))
            log.warning(f"Scryfall rate limit — waiting {wait}s")
            time.sleep(wait)
            raise requests.RequestException("Rate limited")
        if resp.status_code == 200:
            data = resp.json()
            data['colors'] = self._extract_colors(data)
            self._cache[card_name] = data
            return data
        if resp.status_code == 404:
            return self._fuzzy_search(card_name)
        return None

    def _fuzzy_search(self, card_name: str) -> Optional[dict]:
        try:
            resp = requests.get(
                "https://api.scryfall.com/cards/named",
                params={"fuzzy": card_name}, timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                data['colors'] = self._extract_colors(data)
                log.info(f"Fuzzy matched '{card_name}' → '{data.get('name')}'")
                self._cache[card_name] = data
                return data
        except requests.RequestException as e:
            log.warning(f"Fuzzy search failed for '{card_name}': {e}")
        return None

    def enrich_cards(self, card_names: List[str], db: DeckDatabase) -> Tuple[int, int]:
        to_fetch = [n for n in card_names if n not in BASIC_LANDS and not db.card_exists(n)]
        if not to_fetch:
            log.info("All cards already enriched.")
            return 0, 0
        log.info(f"Fetching Scryfall data for {len(to_fetch)} cards…")
        success, failed, failed_names = 0, 0, []
        for i, name in enumerate(to_fetch, 1):
            data = self.fetch_card(name)
            if data:
                db.insert_card_data(name, data)
                success += 1
            else:
                failed += 1
                failed_names.append(name)
            if i % 50 == 0:
                log.info(f"  … {i}/{len(to_fetch)} processed")
            time.sleep(self._REQUEST_DELAY)
        if failed_names:
            log.warning(f"Could not find: {failed_names[:10]}")
        log.info(f"Scryfall enrichment: {success} fetched, {failed} failed")
        return success, failed


# ═════════════════════════════════════════════════════════════════════════════
# SHARED DECK WRITER
# ═════════════════════════════════════════════════════════════════════════════

def write_dck_file(file_hash: str, deck_name: str, card_names: List[str]) -> Path:
    """Write a Forge .dck file. Lands sort to the bottom."""
    FORGE_DECK_DIR.mkdir(parents=True, exist_ok=True)
    file_path = FORGE_DECK_DIR / f"{file_hash}.dck"
    counts = Counter(card_names)
    sorted_cards = sorted(counts.items(), key=lambda kv: (kv[0] in BASIC_LANDS, kv[0]))
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("[metadata]\n")
        f.write(f"Name={deck_name}\n\n[Main]\n")
        for card_name, count in sorted_cards:
            f.write(f"{count} {card_name}\n")
    return file_path


# ═════════════════════════════════════════════════════════════════════════════
# AETHERHUB SCRAPER
# ═════════════════════════════════════════════════════════════════════════════

class AetherHubScraper:
    def __init__(self, db: DeckDatabase):
        self.db = db
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.new_deck_count = 0

    def _generate_hash(self) -> str:
        return uuid.uuid4().hex[:12]

    @retry(max_attempts=3, delay=2, exceptions=(requests.RequestException,))
    def _download_deck(self, slug: str) -> Tuple[bool, List[str]]:
        url  = f"https://aetherhub.com/Deck/Embedded/{slug}"
        resp = self.session.get(url, timeout=15)
        resp.raise_for_status()
        soup  = BeautifulSoup(resp.text, 'html.parser')
        cards = parse_maindeck(soup.get_text())
        return len(cards) > 0, cards

    def scrape_meta_decks(self, num_decks: int = 50):
        log.info(f"── AetherHub: scraping {num_decks} meta decks")
        try:
            resp = self.session.get(AETHERHUB_META_URL, timeout=20)
            resp.raise_for_status()
            soup  = BeautifulSoup(resp.text, 'html.parser')
            links = soup.find_all('a', href=re.compile(r'/Deck/[^/]+-\d+$'))
            exclude = {'deck builder', 'my decks', 'login', 'register'}
            seen: Set[str] = set()
            deck_links: List[Tuple[str, str]] = []
            for link in links:
                href = link.get('href', '')
                name = link.get_text(strip=True)
                if not name or len(name) < 3 or any(p in name.lower() for p in exclude):
                    continue
                m = re.search(r'/Deck/([^/]+-\d+)$', href)
                if not m:
                    continue
                slug = m.group(1)
                if slug not in seen:
                    seen.add(slug)
                    deck_links.append((slug, name))

            log.info(f"   Found {len(deck_links)} meta deck links")

            for i, (slug, name) in enumerate(deck_links[:num_decks]):
                label = f"[AH {i+1}/{min(len(deck_links), num_decks)}]"
                if self.db.exists(slug):
                    log.info(f"{label} {name[:40]} — cached")
                    continue
                tier = 1 if i < 3 else (2 if i < 7 else 3)
                try:
                    success, cards = self._download_deck(slug)
                except Exception as e:
                    log.warning(f"{label} {name[:40]} — failed: {e}")
                    continue
                if not success:
                    log.warning(f"{label} {name[:40]} — 0 cards parsed")
                    continue
                file_hash = self._generate_hash()
                write_dck_file(file_hash, name, cards)
                self.db.add_deck(Deck(
                    slug=slug,
                    url=f"https://aetherhub.com/Deck/{slug}",
                    name=name,
                    colors=colors_from_name(name),
                    is_meta=True, tier=tier, file_hash=file_hash,
                    card_names=cards, tags=['aetherhub', 'meta', 'standard'],
                    last_updated=datetime.now().isoformat(),
                ))
                self.new_deck_count += 1
                log.info(f"{label} {name[:40]} — ✓ ({len(cards)} cards)")
                time.sleep(0.5)

        except Exception as e:
            log.error(f"AetherHub meta scrape failed: {e}", exc_info=True)

    def scrape_user_decks(self, max_pages: int = 4):
        log.info(f"── AetherHub: scraping user decks ({max_pages} pages)")
        try:
            import undetected_chromedriver as uc
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            from selenium.common.exceptions import TimeoutException
        except ImportError as e:
            log.error(f"undetected-chromedriver not installed: {e}")
            return

        options = uc.ChromeOptions()
        for arg in ["--no-sandbox","--disable-dev-shm-usage","--disable-gpu",
                    "--window-size=1920,1080","--disable-blink-features=AutomationControlled"]:
            options.add_argument(arg)
        options.page_load_strategy = 'eager'
        try:
            driver = uc.Chrome(options=options, version_main=145)
            driver.set_page_load_timeout(60)
        except Exception as e:
            log.error(f"Failed to start Chrome: {e}")
            return

        try:
            for page_num in range(1, max_pages + 1):
                log.info(f"   AetherHub user decks — page {page_num}/{max_pages}")
                if page_num == 1:
                    for attempt in range(3):
                        try:
                            driver.get(AETHERHUB_USER_URL)
                            break
                        except Exception:
                            time.sleep(2)
                    time.sleep(5)
                else:
                    try:
                        from selenium.webdriver.support.ui import WebDriverWait
                        from selenium.webdriver.support import expected_conditions as EC
                        WebDriverWait(driver, 10).until(
                            EC.presence_of_element_located(
                                ("id", "metaHubTable_paginate"))
                        )
                        clicked = False
                        try:
                            pl = driver.find_element(
                                "xpath", f"//a[@data-dt-idx='{page_num}']")
                            if "disabled" not in pl.get_attribute("class"):
                                driver.execute_script("arguments[0].scrollIntoView(true);", pl)
                                time.sleep(0.5)
                                driver.execute_script("arguments[0].click();", pl)
                                clicked = True
                        except Exception:
                            pass
                        if not clicked:
                            nb = driver.find_element("id", "metaHubTable_next")
                            if "disabled" not in nb.get_attribute("class"):
                                driver.execute_script("arguments[0].scrollIntoView(true);", nb)
                                time.sleep(0.5)
                                driver.execute_script("arguments[0].click();", nb)
                                clicked = True
                        if not clicked:
                            break
                        time.sleep(5)
                    except Exception as e:
                        log.warning(f"Pagination error on page {page_num}: {e}")
                        continue

                if "challenge" in driver.current_url:
                    time.sleep(8)

                # Wait for the table element to exist in the DOM
                try:
                    WebDriverWait(driver, 20).until(
                        EC.presence_of_element_located(("id", "metaHubTable"))
                    )
                except Exception:
                    continue

                # Scroll up → middle → down → back to table to trigger DataTables
                # JS rendering. Without this, tbody rows may still be empty even
                # though the element is present in the DOM.
                try:
                    driver.execute_script("window.scrollTo(0, 0);")
                    time.sleep(0.5)
                    driver.execute_script(
                        "window.scrollTo(0, document.body.scrollHeight / 2);"
                    )
                    time.sleep(0.5)
                    driver.execute_script(
                        "window.scrollTo(0, document.body.scrollHeight);"
                    )
                    time.sleep(0.5)
                    # Scroll the table itself into view and pause for any
                    # lazy-load / virtualised-row logic to fire
                    driver.execute_script(
                        "document.getElementById('metaHubTable')"
                        ".scrollIntoView({behavior:'smooth', block:'center'});"
                    )
                    time.sleep(1.5)
                except Exception as scroll_err:
                    log.debug(f"Scroll warning (non-fatal): {scroll_err}")

                soup  = BeautifulSoup(driver.page_source, 'html.parser')
                table = soup.find('table', {'id': 'metaHubTable'})
                links = []
                if table:
                    tbody = table.find('tbody')
                    if tbody:
                        links = tbody.find_all('a', href=re.compile(r'/Deck/[^/]+-\d+'))

                seen: Set[str] = set()
                for idx, link in enumerate(links):
                    href = link.get('href', '')
                    if not href or href in seen:
                        continue
                    seen.add(href)
                    m = re.search(r'/Deck/([^/]+-\d+)$', href)
                    if not m:
                        continue
                    slug = m.group(1)
                    name = link.text.strip()
                    label = f"[AH-U {idx+1}]"
                    if self.db.exists(slug):
                        log.debug(f"{label} {name[:35]} — cached")
                        continue
                    try:
                        success, cards = self._download_deck(slug)
                    except Exception as e:
                        log.warning(f"{label} {name[:35]} — error: {e}")
                        continue
                    if not success:
                        continue
                    file_hash = self._generate_hash()
                    write_dck_file(file_hash, name, cards)
                    self.db.add_deck(Deck(
                        slug=slug,
                        url=f"https://aetherhub.com/Deck/{slug}",
                        name=name,
                        colors=colors_from_name(name),
                        is_meta=False, tier=None, file_hash=file_hash,
                        card_names=cards, tags=['aetherhub', 'user', 'standard'],
                        last_updated=datetime.now().isoformat(),
                    ))
                    self.new_deck_count += 1
                    log.info(f"{label} {name[:35]} — ✓ ({len(cards)} cards)")
                    time.sleep(0.3)
        finally:
            driver.quit()

    def write_forge_meta_decks(self):
        """Re-write all AetherHub meta .dck files (idempotent)."""
        written = skipped = 0
        with self.db._connect() as conn:
            rows = conn.execute("""
                SELECT name, file_hash, card_names FROM decks
                WHERE is_meta = 1 AND file_hash IS NOT NULL
                  AND (tags LIKE '%aetherhub%' OR tags NOT LIKE '%goldfish%')
            """).fetchall()
        for name, file_hash, card_names_json in rows:
            if not file_hash or not card_names_json:
                skipped += 1
                continue
            try:
                write_dck_file(file_hash, name, json.loads(card_names_json))
                written += 1
            except Exception as e:
                log.warning(f"Could not write meta deck '{name}': {e}")
                skipped += 1
        log.info(f"Forge meta decks written: {written} (skipped: {skipped})")

    def run(self, meta_count: int = 50, user_pages: int = 4):
        log.info("=" * 60)
        log.info("AETHERHUB SCRAPER")
        self.scrape_meta_decks(meta_count)
        self.scrape_user_decks(user_pages)
        log.info(f"AetherHub total new: {self.new_deck_count}")


# ═════════════════════════════════════════════════════════════════════════════
# MTGGOLDFISH SCRAPER
# ═════════════════════════════════════════════════════════════════════════════

class MTGGoldfishScraper:
    _PAGE_DELAY   = (1.5, 3.0)
    _INITIAL_WAIT = 2.0

    def __init__(self, db: DeckDatabase):
        self.db  = db
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/124.0.0.0 Safari/537.36'
            ),
            'Accept-Language': 'en-US,en;q=0.9',
        })
        self.new_deck_count = 0

    def _generate_hash(self) -> str:
        return uuid.uuid4().hex[:12]

    def _sleep(self):
        lo, hi = self._PAGE_DELAY
        time.sleep(random.uniform(lo, hi))

    @retry(max_attempts=3, delay=2, exceptions=(requests.RequestException,))
    def _get(self, url: str, timeout: int = 20) -> requests.Response:
        resp = self.session.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp

    def _fetch_deck_links(self) -> List[Tuple[str, str, int]]:
        log.info(f"── MTGGoldfish: fetching metagame index")
        resp = self._get(GOLDFISH_META_URL)
        soup = BeautifulSoup(resp.text, 'html.parser')
        time.sleep(self._INITIAL_WAIT)

        results: List[Tuple[str, str, int]] = []
        seen: Set[str] = set()

        for rank, span in enumerate(
            soup.find_all('span', {'class': 'deck-price-paper'})[1:]
        ):
            anchor = span.find('a')
            if not anchor:
                continue
            name = re.sub(r'\s+', ' ', span.get_text(separator=' ', strip=True))
            if not name or name == 'Other' or name in seen:
                continue
            seen.add(name)
            href = anchor.get('href', '')
            if not href.startswith('/archetype/') and not href.startswith('/deck/'):
                continue
            deck_url = GOLDFISH_BASE_URL + href
            tier = 1 if rank < 3 else (2 if rank < 7 else 3)
            results.append((name, deck_url, tier))

        log.info(f"   Found {len(results)} decks on metagame page")
        return results

    @retry(max_attempts=3, delay=2, exceptions=(requests.RequestException,))
    def _download_deck(self, deck_url: str) -> List[str]:
        resp = self._get(deck_url)
        soup = BeautifulSoup(resp.text, 'html.parser')
        inp  = soup.find('input', {'id': 'deck_input_deck'})
        if not inp or not inp.get('value'):
            log.warning(f"No deck_input_deck at {deck_url}")
            return []
        raw = re.sub(r'(?i)\bsideboard\b', 'Sideboard', inp['value'])
        return parse_maindeck(raw)

    def run(self, max_decks: int = 60):
        log.info("=" * 60)
        log.info("MTGGOLDFISH SCRAPER")
        deck_links = self._fetch_deck_links()
        if not deck_links:
            log.warning("No deck links found — page structure may have changed.")
            return

        total = min(len(deck_links), max_decks)
        for i, (name, url, tier) in enumerate(deck_links[:max_decks]):
            url_slug = re.sub(r'[^a-z0-9]+', '-',
                              url.split('/')[-1].split('#')[0].lower()).strip('-')
            slug  = f"{GOLDFISH_PREFIX}{url_slug}"
            label = f"[GF {i+1}/{total}]"

            if self.db.exists(slug):
                log.info(f"{label} {name[:45]} — cached")
                continue

            log.info(f"{label} {name[:45]} …")
            try:
                cards = self._download_deck(url)
            except Exception as e:
                log.warning(f"  ✗ {e}")
                self._sleep()
                continue

            if not cards:
                log.warning(f"  ✗ 0 cards parsed")
                self._sleep()
                continue

            file_hash = self._generate_hash()
            write_dck_file(file_hash, name, cards)
            self.db.add_deck(Deck(
                slug=slug, url=url, name=name,
                colors=colors_from_name(name),
                is_meta=True, tier=tier, file_hash=file_hash,
                card_names=cards, tags=['goldfish', 'standard'],
                last_updated=datetime.now().isoformat(),
            ))
            self.new_deck_count += 1
            log.info(f"  ✓ {len(cards)} cards  (tier {tier})")
            self._sleep()

        log.info(f"MTGGoldfish total new: {self.new_deck_count}")


# ═════════════════════════════════════════════════════════════════════════════
# UNIFIED RUNNER
# ═════════════════════════════════════════════════════════════════════════════

class MTGScraper:
    """
    Top-level orchestrator.  Instantiate with a source and call run().
    source: 'aetherhub' | 'mtggoldfish' | 'all'
    """

    def __init__(self, source: str = 'all', db_path: Path = DB_PATH):
        self.source = source.lower().strip()
        self.db     = DeckDatabase(db_path)
        FORGE_DECK_DIR.mkdir(parents=True, exist_ok=True)

    def _enrich(self):
        """Single shared Scryfall enrichment pass after all scraping is done."""
        log.info("── Scryfall enrichment pass")
        all_cards = self.db.get_all_card_names()
        to_fetch  = [c for c in all_cards if c not in BASIC_LANDS]
        missing   = self.db.get_missing_cards(to_fetch)
        if not missing:
            log.info("   All cards already enriched.")
            return
        ScryfallEnricher().enrich_cards(missing, self.db)

    def backfill(self, fix_colors: bool = False):
        if fix_colors:
            removed = self.db.reset_empty_colors()
            log.info(f"Removed {removed} cards with empty colors for re-fetch")
        log.info("── Backfilling missing card data")
        all_cards = self.db.get_all_card_names()
        to_fetch  = [c for c in all_cards if c not in BASIC_LANDS]
        missing   = self.db.get_missing_cards(to_fetch)
        if not missing:
            log.info("   Nothing to backfill.")
            return
        log.info(f"   {len(missing)} cards need data")
        ScryfallEnricher().enrich_cards(missing, self.db)

    def run(
        self,
        meta_count:  int  = 50,
        user_pages:  int  = 4,
        max_decks:   int  = 60,
        skip_enrich: bool = False,
        force:       bool = False,
    ):
        if force:
            with self.db._connect() as conn:
                count = conn.execute("SELECT COUNT(*) FROM decks").fetchone()[0]
                conn.execute("DELETE FROM decks")
            log.info(f"Force-cleared {count} decks (card data preserved)")

        log.info("=" * 60)
        log.info(f"MTG SCRAPER  |  source={self.source}  |  DB={self.db.db_path}")
        counts_before = self.db.get_count_by_source()
        log.info(
            f"DB before — aetherhub:{counts_before['aetherhub']}  "
            f"goldfish:{counts_before['goldfish']}  "
            f"other:{counts_before['other']}  "
            f"total:{self.db.get_count()}"
        )
        log.info("=" * 60)

        run_ah = self.source in ('aetherhub', 'all')
        run_gf = self.source in ('mtggoldfish', 'all')

        if run_ah:
            ah = AetherHubScraper(self.db)
            ah.run(meta_count=meta_count, user_pages=user_pages)

        if run_gf:
            gf = MTGGoldfishScraper(self.db)
            gf.run(max_decks=max_decks)

        if not skip_enrich:
            self._enrich()

        # Always re-write Forge files so newly-added decks are picked up
        if run_ah:
            AetherHubScraper(self.db).write_forge_meta_decks()

        counts_after = self.db.get_count_by_source()
        log.info("=" * 60)
        log.info("DONE")
        log.info(
            f"DB after  — aetherhub:{counts_after['aetherhub']}  "
            f"goldfish:{counts_after['goldfish']}  "
            f"other:{counts_after['other']}  "
            f"total:{self.db.get_count()}"
        )
        log.info("=" * 60)


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Unified MTG deck scraper — AetherHub + MTGGoldfish',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 mtg_scraper.py                          # scrape both sources
  python3 mtg_scraper.py --source aetherhub       # AetherHub only
  python3 mtg_scraper.py --source mtggoldfish     # MTGGoldfish only
  python3 mtg_scraper.py --source all --meta 75 --pages 6 --max-decks 80
  python3 mtg_scraper.py --backfill-only
  python3 mtg_scraper.py --backfill-only --fix-colors
        """,
    )

    parser.add_argument(
        '--source', default='all',
        choices=['all', 'aetherhub', 'mtggoldfish'],
        help='Which source(s) to scrape (default: all)',
    )
    parser.add_argument('--meta',       type=int, default=50,
                        help='AetherHub: number of meta decks (default: 50)')
    parser.add_argument('--pages',      type=int, default=4,
                        help='AetherHub: user-deck pages (default: 4)')
    parser.add_argument('--max-decks',  type=int, default=60,
                        help='MTGGoldfish: max decks to download (default: 60)')
    parser.add_argument('--skip-enrich', action='store_true',
                        help='Skip Scryfall enrichment after scraping')
    parser.add_argument('--backfill-only', action='store_true',
                        help='Only backfill missing card data — no scraping')
    parser.add_argument('--fix-colors',  action='store_true',
                        help='Re-fetch cards with empty color data (use with --backfill-only)')
    parser.add_argument('--force',       action='store_true',
                        help='Clear all decks and re-scrape from scratch (card data preserved)')
    parser.add_argument('--db',          default=str(DB_PATH),
                        help=f'Path to SQLite DB (default: {DB_PATH})')

    args = parser.parse_args()

    scraper = MTGScraper(source=args.source, db_path=Path(args.db))

    if args.backfill_only:
        scraper.backfill(fix_colors=args.fix_colors)
    else:
        scraper.run(
            meta_count  = args.meta,
            user_pages  = args.pages,
            max_decks   = args.max_decks,
            skip_enrich = args.skip_enrich,
            force       = args.force,
        )
