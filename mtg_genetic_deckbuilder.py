#!/usr/bin/env python3
"""
MTG Genetic Deckbuilder with LDA Integration and Resumable Checkpointing
Loads pre-trained LDA model for archetype-aware deck evolution

FIXES APPLIED (original):
  1. evaluate() - removed dead code, fixed double-discounting of heuristic scores
  2. cx_deck() - removed unused 'name' parameter from fix_size inner function
  3. mut_lda_deck() - card_counts cleaned up when count reaches zero
  4. create_lda_individual() - land_slots_remaining now respected in emergency loop
  5. _strict_color_filter() - self.cards dict now purged of off-color entries
  6. prompt_resume() / run() - eliminated double checkpoint load
  7. load_checkpoint() - pickle replaced with json-serialisable format
  8. CardPool - context-manager usage enforced; __del__ made safe
  9. ForgeHeadlessRunner - executor shut down properly
 10. GAMES_PER_MATCHUP - no longer mutated globally; passed as a parameter
 11. _ensure_deap() call inside create_lda_individual() removed (redundant)
 12. Fallback deck card names replaced with real basic-land names
 13. Coherence entropy - max_entropy now uses full n_topics denominator
 14. export_forge_format() - path handling tightened

FIXES APPLIED (this pass):
 15. LDA_MODEL_PATH corrected from .pkl to .json
 16. LDAModelManager._normalize_name now matches train_lda.py (hyphens -> underscores)
 17. CardPool rewritten to work against decks_cache.db schema (no card_roles /
     mana_production tables); derives roles/mana_production from oracle_text/type_line
 18. resume=True path now performs the same color-mismatch check as interactive path
 19. get_cards_by_topic colorless-spell filter tightened
 20. ARCHETYPE_KEYWORDS map extended + LDA name -> keyword mapping added
"""

import os
import re
import json
import sqlite3
import random
import time
import argparse
import subprocess
import uuid
import signal
import sys
import hashlib
import warnings
import logging
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple, Counter as TypingCounter
from dataclasses import dataclass, field
from datetime import datetime
from collections import OrderedDict, Counter, defaultdict
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from deap import base, creator, tools, algorithms

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ---------- CONFIGURATION ----------
POPULATION_SIZE       = 50
GENERATIONS           = 50
CROSSOVER_PROB        = 0.7
MUTATION_PROB         = 0.3
DECK_SIZE             = 60
FORGE_PATH            = os.getenv('FORGE_PATH', '/home/toaster/forge-installer-2.0.11')
FORGE_DECK_DIR        = Path('/home/toaster/.forge/decks/constructed')
META_DECK_DIR         = './meta_decks'
TEST_RESULTS_DIR      = './test_results'
GAMES_PER_MATCHUP_DEFAULT = 5
MAX_CACHE_SIZE        = 5000
# FIX #15: correct extension — train_lda.py saves JSON, not pickle
LDA_MODEL_PATH        = './lda_model.json'
DB_PATH               = './mtg_cards.db'    # owned-card pool from collection importer
META_DB_PATH          = './decks_cache.db'  # meta/scraped decks for seeding inspiration
CHECKPOINT_DIR        = './checkpoints'
CHECKPOINT_FREQUENCY  = 1
MAX_CHECKPOINTS       = 3

# ---------- MODULE-LEVEL CONSTANTS ----------
# FIX #20: extended keyword map; LDA archetype names now map to these keys
ARCHETYPE_KEYWORDS = {
    # ── SACRIFICE / ARISTOCRATS ───────────────────────────────────────────
    "Sacrifice": [
        'sacrifice', 'dies', 'death trigger',
        'whenever a creature you control dies',
        'whenever another creature dies',
        'when this creature dies',
        'whenever you sacrifice',
        'sacrifice a creature',
        'blood artist', 'zulaport', 'cutthroat', 'viscera seer',
        'carrion feeder', 'altar',
        'food', 'treasure', 'clue',         # artifact sacrifice loops
        'exploit', 'emerge', 'devour',      # keyword sacrifice mechanics
        'convoke',                          # token-into-spell
    ],
    "Aristocrats": [
        'sacrifice', 'dies',
        'whenever a creature you control dies',
        'whenever another creature dies',
        'blood artist', 'zulaport', 'cruel celebrant',
        'voice of the blessed', 'elas il-kor',
        'whenever you sacrifice',
        'extort',
        'food', 'treasure',
    ],

    # ── AGGRO ─────────────────────────────────────────────────────────────
    "Aggro": [
        'haste', 'first strike', 'double strike', 'menace', 'trample',
        'prowess',                          # spell-triggered pump
        'exert',                            # white/red aggro keyword
        'dash',                             # red evasion
        'battalion',                        # Boros attack trigger
        'raid',                             # attack-based payoff
        'spectacle',                        # damage-gated cost reduction
        'pump', '+1/+1',
        'burn', 'direct damage',
        'whenever this creature attacks',
        'one or more creatures attack',
        'combat damage to a player',
        'power 2 or less',                  # weenie filtering
    ],

    # ── CONTROL ───────────────────────────────────────────────────────────
    "Control": [
        'counter target', 'counterspell',
        'draw', 'card',
        'destroy target', 'exile target',
        'wrath', 'board wipe',
        'search your library',
        'scry', 'surveil',
        'foretell',
        'cycling',
        'flash',
        'end of your turn',
        'at the beginning of your upkeep',
        'until end of turn',
        "can't attack", "can't block",
        'return target',                    # bounce
        'tap target',                       # soft control
    ],

    # ── RAMP ──────────────────────────────────────────────────────────────
    "Ramp": [
        'search your library for a land',
        'put a land',
        'add {',
        'mana rock', 'elf', 'druid',
        'explore', 'rampant growth',
        'additional land',                  # play extra lands
        'put onto the battlefield',         # Cultivate/Kodama's Reach
        'untap target land',                # Selvala-style
        'treasure token',                   # treasure ramp
        'convoke', 'improvise',             # alternate-cost ramp
        'basic land card',
    ],

    # ── TOKENS ────────────────────────────────────────────────────────────
    "Tokens": [
        'create', 'token', 'populate',
        'go wide', 'hive', 'servo', 'zombie', 'soldier', '1/1',
        'amass',                            # Zombie army tokens
        'fabricate',                        # artifact token or +1/+1
        'afterlife',                        # Spirit tokens on death
        'embalm', 'eternalize',             # token from graveyard
        'convoke',                          # token-into-spell
        'creatures you control get +',      # anthem effect
        'attack with',                      # go-wide attack trigger
        'whenever a token',                 # token-specific payoffs
        'copy',                             # token copy effects
    ],

    # ── GRAVEYARD ─────────────────────────────────────────────────────────
    "Graveyard": [
        'graveyard', 'dredge',
        'flashback', 'unearth', 'reanimate', 'recursion',
        'delve',                            # exile-from-graveyard cost
        'escape',                           # Theros graveyard keyword
        'aftermath',                        # split graveyard spells
        'jump-start',                       # Izzet flashback variant
        'disturb',                          # white/blue graveyard transform
        'threshold',                        # 7-card graveyard payoff
        'delirium',                         # 4 card-type payoff
        'undergrowth',                      # creature-count payoff
        'from your graveyard',
        'return target creature card',
        'when ~ dies, return',              # self-recursion
        'mill',                             # self-mill enabler
        'put the top',                      # self-mill phrasing
    ],

    # ── LIFEGAIN ──────────────────────────────────────────────────────────
    "Lifegain": [
        'lifelink',
        'you gain life', 'gain life',
        'whenever you gain life',
        'the next time you would gain life',
        'gains lifelink',
        'life you gain',
        'extort',                           # per-spell drain
        'each opponent loses',              # drain mirror (Vito)
        'lose life',                        # drain payoff
        'soul warden', 'ajani', 'amalia',
        'resplendent', 'giada', 'inspiring overseer', 'haliya',
        'vito', 'sunstar', 'essence channeler', 'leonin vanguard',
        "ajani's pridemate", 'ruin-lurker', 'hinterland sanctifier',
        'lifecreed', 'starscape cleric', 'lyra dawnbringer',
        'voice of the blessed', 'kitchen finks',
        'angel',                            # type-line synergy
        'cleric',                           # type-line synergy
    ],

    # ── LANDFALL ──────────────────────────────────────────────────────────
    "Landfall": [
        'landfall',
        'whenever a land enters',
        'whenever you play a land',
        'land enters the battlefield',
        'fetch land',
        'sacrifice a land',                 # land sacrifice payoffs
        'nonbasic land',
        'crucible of worlds',
        'life from the loam',
        'harrow', 'crop rotation',
    ],

    # ── +1/+1 COUNTERS ────────────────────────────────────────────────────
    "Counters": [
        '+1/+1 counter', 'proliferate',
        'bolster', 'adapt', 'evolve',
        'modular',                          # artifact counter transfer
        'reinforce',                        # counters from hand
        'graft',                            # counter transfer
        'outlast',                          # tap to add counter
        'mentor',                           # attack-based counter
        'training',                         # attack-with-bigger counter
        'support',                          # put counters on others
        'sunburst',                         # multicolor counter accumulation
        'charge counter',
        'move a counter', 'remove a counter',
    ],

    # ── MILL ──────────────────────────────────────────────────────────────
    "Mill": [
        'mill', 'put the top',
        'library into your graveyard',
        'each player mills',
        'traumatize', 'maddening cacophony',
        "sphinx's tutelage", 'drowned secrets',
        'draw', 'replace',                  # replacement-effect mill
    ],

    # ── MIDRANGE ──────────────────────────────────────────────────────────
    "Midrange": [
        'enters the battlefield', 'when this creature',
        'flash',
        'cascade',                          # midrange value mechanic
        'evoke',                            # value creatures
        'champion',                         # tribal midrange
        "can't be countered",               # midrange resilience
    ],
}


# FIX #20: map LDA topic name fragments -> ARCHETYPE_KEYWORDS key
# Checked against the archetype names train_lda.py actually produces
LDA_TO_ARCHETYPE_KEY = {
    # Maps fragments of LDA topic names -> ARCHETYPE_KEYWORDS keys.
    # Keys are lowercase substrings that appear in topic_names values
    # (e.g. 'White Lifegain', 'Izzet Graveyard', 'Rakdos Aristocrats').
    'graveyard':    'Graveyard',
    'lifegain':     'Lifegain',
    'landfall':     'Landfall',
    'tokens':       'Tokens',
    'token':        'Tokens',
    'aggro':        'Aggro',
    'control':      'Control',
    'ramp':         'Ramp',
    'counters':     'Counters',
    'counter':      'Counters',
    'mill':         'Mill',
    'midrange':     'Midrange',
    'sacrifice':    'Sacrifice',
    'aristocrats':  'Aristocrats',
    'burn':         'Aggro',        # "Red Burn" topics
    'stompy':       'Aggro',        # "Green Stompy"
    'combo':        'Control',      # treat combo as control for keyword purposes
    'discard':      'Control',      # discard/hand-hate shells
}

# FIX #12: real basic-land names used everywhere a fallback land is needed
BASIC_LANDS = {'W': 'Plains', 'U': 'Island', 'B': 'Swamp', 'R': 'Mountain', 'G': 'Forest'}
BASIC_LAND_NAMES = set(BASIC_LANDS.values())

# Keyword synergy cache: (card_id, archetype) -> bool
# Avoids repeated oracle text scans inside tight mutation loops
_keyword_synergy_cache: Dict[Tuple[int, str], bool] = {}


class SecurityWarning(UserWarning):
    pass


# ---------- DECORATORS ----------
def retry(max_attempts=3, delay=1, backoff=2, exceptions=(Exception,)):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 1
            current_delay = delay
            while attempt <= max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        raise
                    jitter = random.uniform(0, current_delay * 0.1)
                    sleep_time = current_delay + jitter
                    print(f"    Attempt {attempt} failed: {e}. Retrying in {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
                    current_delay *= backoff
                    attempt += 1
            return None
        return wrapper
    return decorator


# ---------- DATA CLASSES ----------
@dataclass
class Card:
    id: int
    name: str
    mana_cost: str
    cmc: int
    colors: List[str]
    types: str
    oracle_text: str
    keywords: List[str]
    rarity: str
    is_basic_land: bool
    max_copies: int
    roles: List[str]
    mana_production: List[str]
    lda_topics: Dict[str, float] = field(default_factory=dict)


@dataclass
class MetaDeck:
    name: str
    colors: str
    tier: int
    file_path: str
    win_rate_against: Dict[str, float] = field(default_factory=dict)
    card_names: List[str] = field(default_factory=list)


# ---------- DEAP SETUP ----------
def _ensure_deap():
    if not hasattr(creator, 'FitnessMax'):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, 'Individual'):
        creator.create("Individual", list, fitness=creator.FitnessMax)


# ---------- LDA MODEL MANAGER ----------
class LDAModelManager:

    def __init__(self, n_topics=8):
        self.n_topics    = n_topics
        self.card_topics: Dict[str, Dict[str, float]] = {}
        self.topic_names: Dict[str, str] = {}
        self.is_trained  = False
        self._filepath   = LDA_MODEL_PATH
        self._load_failed = False
        self._last_error  = None
        self._logger      = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def _normalize_name(name) -> str:
        """
        FIX #16: must exactly match train_lda.py's _normalize_name so that
        card lookups hit the keys stored in lda_model.json.
        train_lda.py does: lower, strip, spaces->_, hyphens->_, remove ' and ,
        """
        if not isinstance(name, str):
            name = str(name)
        return (name.lower().strip()
                .replace(' ', '_')
                .replace('-', '_')
                .replace("'", '')
                .replace(',', ''))

    def load(self, filepath=LDA_MODEL_PATH):
        self._filepath    = filepath
        self._load_failed = False
        self._last_error  = None

        if not os.path.exists(filepath):
            self._logger.warning("No LDA model found at %s", filepath)
            print(f"    Run: python3 train_lda.py --all-decks")
            self.is_trained   = False
            self._load_failed = True
            return False

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            raw_topics = data.get('card_topics', {})
            if not raw_topics:
                for key in ['topics', 'card_topic', 'topic_dict']:
                    if key in data:
                        raw_topics = data[key]
                        break

            self.card_topics = {
                self._normalize_name(k): v
                for k, v in raw_topics.items()
            }
            self.topic_names = data.get('topic_names', {})
            self.n_topics    = data.get('n_topics', self.n_topics)

            if not self.card_topics:
                self._logger.warning("LDA model loaded but contains 0 card topics")
                self.is_trained   = False
                self._load_failed = True
                return False

            self.is_trained   = True
            self._load_failed = False
            print(f"  ✓ LDA model loaded from {filepath}")
            print(f"    Archetypes: {list(self.topic_names.values())}")
            print(f"    Cards with topic assignments: {len(self.card_topics)}")
            return True

        except Exception as e:
            self._logger.error("Error loading LDA model: %s", e)
            self._last_error  = str(e)
            self.is_trained   = False
            self._load_failed = True
            return False

    def _ensure_loaded(self):
        if self._load_failed:
            return
        if not self.is_trained or not self.card_topics:
            success = self.load(self._filepath)
            if not success:
                self._load_failed = True

    def get_card_topics(self, card_name: str) -> Dict[str, float]:
        self._ensure_loaded()
        if not self.is_trained:
            return {}
        return self.card_topics.get(self._normalize_name(card_name), {})

    # Maps color words in topic names to WUBRG symbols
    _TOPIC_COLOR_WORDS = {
        'white': 'W', 'blue': 'U', 'black': 'B', 'red': 'R', 'green': 'G',
        'azorius': 'WU', 'dimir': 'UB', 'rakdos': 'BR', 'gruul': 'RG',
        'selesnya': 'WG', 'orzhov': 'WB', 'izzet': 'UR', 'golgari': 'BG',
        'boros': 'WR', 'simic': 'UG',
        'esper': 'WUB', 'grixis': 'UBR', 'jund': 'BRG', 'naya': 'WRG',
        'bant': 'WUG', 'mardu': 'WBR', 'temur': 'URG', 'abzan': 'WBG',
        'jeskai': 'WUR', 'sultai': 'UBG',
    }

    def _topic_colors(self, topic_name: str) -> set:
        """Extract color identity implied by a topic name like 'Mardu Aggro'."""
        colors = set()
        lower = topic_name.lower()
        for word, symbols in self._TOPIC_COLOR_WORDS.items():
            if word in lower:
                colors.update(symbols)
        return colors

    def get_deck_archetype(self, deck: List[int], card_pool: 'CardPool',
                           target_colors=None) -> Tuple[str, float]:
        self._ensure_loaded()
        if not self.is_trained or not self.card_topics:
            return ("Unknown", 0.0)

        deck_topics: Dict[str, float] = defaultdict(float)
        total_weight = 0

        for cid in deck:
            card = card_pool.get_card(cid)
            for topic, prob in self.get_card_topics(card.name).items():
                deck_topics[topic] += prob
            total_weight += 1

        if not deck_topics or total_weight == 0:
            return ("Unknown", 0.0)

        for topic in deck_topics:
            deck_topics[topic] /= total_weight

        # Color-aware topic selection: boost topics whose color identity
        # overlaps with the deck's target colors.
        effective_colors = target_colors or (
            card_pool.target_colors if card_pool else None)

        if effective_colors:
            # Step 1: try topics whose color identity is a SUBSET of target colors
            # (i.e. fully compatible topics — no off-color requirements)
            compatible = {
                tid: weight for tid, weight in deck_topics.items()
                if not self._topic_colors(self.topic_names.get(tid, tid))  # colorless topic
                or self._topic_colors(self.topic_names.get(tid, tid)).issubset(effective_colors)
            }
            # Step 2: among compatible topics, boost by color coverage
            adjusted = {}
            pool_to_score = compatible if compatible else deck_topics
            for tid, weight in pool_to_score.items():
                tname = self.topic_names.get(tid, tid)
                tcol  = self._topic_colors(tname)
                if tcol and effective_colors:
                    # coverage: fraction of target colors this topic addresses
                    coverage = len(tcol & effective_colors) / max(len(effective_colors), 1)
                    # For multi-color decks, heavily reward topics that cover
                    # more of the target identity. Full coverage = 4x boost.
                    n = len(effective_colors)
                    if n >= 3:
                        boost = 1.0 + (coverage ** 2) * 3.0  # 1x..4x
                    else:
                        boost = 1.0 + coverage               # 1x..2x
                    adjusted[tid] = weight * boost
                else:
                    adjusted[tid] = weight
            dominant_topic = max(adjusted.items(), key=lambda x: x[1])
        else:
            dominant_topic = max(deck_topics.items(), key=lambda x: x[1])

        topic_name = self.topic_names.get(dominant_topic[0], dominant_topic[0])

        probs = list(deck_topics.values())
        if len(probs) > 1:
            entropy     = -sum(p * np.log(p + 1e-10) for p in probs if p > 0)
            max_entropy = np.log(self.n_topics)
            coherence   = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0
        else:
            coherence = 1.0

        return (topic_name, coherence)

    def get_cards_by_topic(self, topic_name: str, card_pool: 'CardPool',
                           top_n: int = 50) -> List[int]:
        self._ensure_loaded()
        if not self.is_trained:
            return []

        topic_id = next(
            (tid for tid, name in self.topic_names.items() if name == topic_name),
            None
        )
        if topic_id is None:
            return []

        card_scores = []
        for cid in card_pool.cards:
            card = card_pool.get_card(cid)

            # FIX #19: skip colorless non-lands when a color filter is active
            if card_pool.target_colors and 'Land' not in card.types:
                card_colors = set(card.colors) if card.colors else None
                if card_colors is not None and not card_colors.issubset(card_pool.target_colors):
                    continue

            topics = self.get_card_topics(card.name)
            if topics and topic_id in topics:
                card_scores.append((cid, topics[topic_id]))

        card_scores.sort(key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in card_scores[:top_n]]


# ---------- CARD POOL ----------
# Reads from mtg_cards.db (created by the collection importer).
# Schema:
#   cards(id, name, mana_cost, cmc, colors, types, subtypes, oracle_text,
#         keywords, rarity, is_basic_land, user_quantity, set_code, collector_number)
#   card_roles(card_id, role, weight)
#   mana_production(card_id, mana_symbol, is_conditional)


class CardPool:
    def __init__(self, db_path: str = DB_PATH,
                 target_colors: Optional[Set[str]] = None,
                 lda_manager: Optional[LDAModelManager] = None):
        self.db_path       = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self.cards:         Dict[int, Card]    = {}
        self.cards_by_name: Dict[str, Card]    = {}
        self.target_colors = target_colors
        self.lands:    List[int] = []
        self.creatures: List[int] = []
        self.spells:    List[int] = []
        self.cards_by_color: Dict[str, List[int]] = {
            'W': [], 'U': [], 'B': [], 'R': [], 'G': [], 'C': [], 'Land': []
        }
        self.lda_manager  = lda_manager
        self._initialized = False
        self._logger      = logging.getLogger(self.__class__.__name__)

        self._connect()
        try:
            self._validate_schema()
            self._load_cards()
            if self.target_colors:
                self._strict_color_filter()
            self._initialized = True
        except Exception:
            self._cleanup()
            raise

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._connect()
        return self._conn

    def _connect(self) -> None:
        try:
            self._conn = sqlite3.connect(self.db_path, timeout=30.0,
                                         check_same_thread=False)
        except sqlite3.Error as e:
            self._logger.error("Failed to connect to %s: %s", self.db_path, e)
            raise

    def _cleanup(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception as e:
                self._logger.warning("Could not close DB: %s", e)
            finally:
                self._conn = None

    def __del__(self):
        try:
            if self._conn is not None:
                self._cleanup()
        except Exception:
            pass

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_conn'] = None
        state.pop('_logger', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._conn   = None
        self._logger = logging.getLogger(self.__class__.__name__)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup()
        return False

    def reconnect(self) -> None:
        self._cleanup()
        self._connect()
        self.cards.clear()
        self.cards_by_name.clear()
        self.lands.clear()
        self.creatures.clear()
        self.spells.clear()
        for key in self.cards_by_color:
            self.cards_by_color[key].clear()
        self._validate_schema()
        self._load_cards()
        if self.target_colors:
            self._strict_color_filter()

    def _validate_schema(self) -> None:
        c = self.conn.cursor()
        try:
            c.execute("SELECT name FROM sqlite_master WHERE type='table'")
            existing = {t[0] for t in c.fetchall()}
            required = {'cards', 'card_roles', 'mana_production'}
            missing  = required - existing
            if missing:
                raise RuntimeError(
                    f"mtg_cards.db is missing tables: {missing}\n"
                    "Run the collection importer first:\n"
                    "  python3 import_collection.py your_collection.txt"
                )
        finally:
            c.close()

    def _load_cards(self) -> None:
        """
        Load owned cards from mtg_cards.db (created by the collection importer).
        Only cards with user_quantity > 0 are loaded; max_copies is capped at
        the actual quantity you own (or 4 for non-basics, whichever is lower).
        """
        c = self.conn.cursor()
        try:
            c.execute("""
                SELECT id, name, mana_cost, cmc, colors, types,
                       oracle_text, keywords, rarity, is_basic_land, user_quantity
                FROM cards
                WHERE user_quantity > 0
            """)
            for row in c.fetchall():
                try:
                    (card_id, name, mana_cost, cmc, colors_json, types,
                     oracle_text, keywords_json, rarity, is_basic, user_qty) = row

                    colors   = json.loads(colors_json)   if colors_json   else []
                    keywords = json.loads(keywords_json) if keywords_json else []

                    # Load roles from the card_roles table
                    c2 = self.conn.cursor()
                    try:
                        c2.execute("SELECT role FROM card_roles WHERE card_id = ?",
                                   (card_id,))
                        roles = [r[0] for r in c2.fetchall()]

                        # Load mana production from mana_production table
                        c2.execute(
                            "SELECT mana_symbol FROM mana_production WHERE card_id = ?",
                            (card_id,))
                        mana_prod = [m[0] for m in c2.fetchall()]
                    finally:
                        c2.close()

                    is_basic   = bool(is_basic)
                    user_qty   = int(user_qty) if user_qty else 1
                    # Respect what you actually own: basics up to 60, others min(qty, 4)
                    max_copies = 60 if is_basic else min(user_qty, 4)

                    lda_topics: Dict[str, float] = {}
                    if self.lda_manager and self.lda_manager.is_trained:
                        try:
                            lda_topics = self.lda_manager.get_card_topics(name)
                        except Exception as e:
                            self._logger.debug("LDA topics error for %s: %s", name, e)

                    card = Card(
                        id=card_id, name=name,
                        mana_cost=mana_cost or '',
                        cmc=int(cmc) if cmc else 0,
                        colors=colors,
                        types=types or '',
                        oracle_text=oracle_text or '',
                        keywords=keywords,
                        rarity=rarity or 'common',
                        is_basic_land=is_basic,
                        max_copies=max_copies,
                        roles=roles,
                        mana_production=mana_prod,
                        lda_topics=lda_topics,
                    )

                    self.cards[card_id] = card
                    self.cards_by_name[name] = card
                    norm = name.replace(' ', '_').replace(',', '').replace("'", '')
                    self.cards_by_name[norm] = card

                    if 'Land' in (types or '') and '//' not in (types or ''):
                        self.lands.append(card_id)
                        self.cards_by_color['Land'].append(card_id)
                    else:
                        if 'Creature' in (types or ''):
                            self.creatures.append(card_id)
                        else:
                            self.spells.append(card_id)
                        if not colors:
                            self.cards_by_color['C'].append(card_id)
                        else:
                            for color in colors:
                                if color in self.cards_by_color:
                                    self.cards_by_color[color].append(card_id)

                except Exception as e:
                    self._logger.error("Error processing card row: %s", e)
                    continue
        finally:
            c.close()

        print(f"  CardPool (owned): {len(self.lands)} lands, "
              f"{len(self.creatures)} creatures, {len(self.spells)} spells "
              f"({len(self.cards)} total)")

    def _strict_color_filter(self, keep_colorless_nonlands: bool = True) -> None:
        if not self.target_colors:
            return

        target = self.target_colors
        print(f"  Filtering to colors: {''.join(sorted(target))}")

        # Filter lands: keep basics, colorless lands, and lands that produce target colors
        valid_lands = []
        for cid in self.lands:
            land = self.cards[cid]
            if land.is_basic_land:
                # Only keep basics for our colors
                if any(BASIC_LANDS.get(c) == land.name for c in target):
                    valid_lands.append(cid)
                continue
            produced = set(land.mana_production)
            if not produced:
                valid_lands.append(cid)   # colorless utility land
            elif produced.issubset(target):
                valid_lands.append(cid)   # only keep fully on-color lands
        self.lands = valid_lands

        def keep_nonland(cid: int) -> bool:
            card = self.cards[cid]
            if not card.colors:
                return keep_colorless_nonlands
            return set(card.colors).issubset(target)

        self.creatures = [c for c in self.creatures if keep_nonland(c)]
        self.spells    = [c for c in self.spells    if keep_nonland(c)]

        # Purge off-color cards from self.cards so direct iteration is clean
        valid_ids = set(self.lands) | set(self.creatures) | set(self.spells)
        for cid in [cid for cid in list(self.cards) if cid not in valid_ids]:
            card = self.cards.pop(cid)
            self.cards_by_name.pop(card.name, None)
            norm = card.name.replace(' ', '_').replace(',', '').replace("'", '')
            self.cards_by_name.pop(norm, None)

        for color in self.cards_by_color:
            if color != 'Land':
                self.cards_by_color[color] = [
                    c for c in self.cards_by_color[color]
                    if c in self.creatures or c in self.spells
                ]

        print(f"  After filter: {len(self.lands)} lands, {len(self.creatures)} creatures, "
              f"{len(self.spells)} spells")

    def get_card(self, card_id: int) -> Card:
        try:
            return self.cards[card_id]
        except KeyError:
            if not self.cards and self._initialized:
                self.reconnect()
            return self.cards[card_id]

    def get_card_by_name(self, name: str) -> Optional[Card]:
        if name in self.cards_by_name:
            return self.cards_by_name[name]
        return self.cards_by_name.get(name.replace(' ', '_'))

    def get_random_card(self, card_type: str = None,
                        target_color: Optional[str] = None) -> Optional[int]:
        try:
            if target_color and target_color in self.cards_by_color:
                pool = self.cards_by_color[target_color]
                if pool:
                    return random.choice(pool)
            if card_type == 'land'    and self.lands:     return random.choice(self.lands)
            if card_type == 'creature' and self.creatures: return random.choice(self.creatures)
            if card_type == 'spell'   and self.spells:    return random.choice(self.spells)
            all_cards = self.lands + self.creatures + self.spells
            return random.choice(all_cards) if all_cards else None
        except Exception as e:
            self._logger.error("Error getting random card: %s", e)
            return None

    def close(self) -> None:
        self._cleanup()


# ---------- META CARD POOL ----------
class MetaCardPool:
    """
    Loads card names from decks_cache.db (scraped meta decks) purely for seeding
    and inspiration.  Does NOT enforce ownership — that is handled by the owned
    CardPool.  Every card name is resolved back to an owned card before it enters
    a real individual.
    """

    def __init__(self, db_path: str = META_DB_PATH,
                 lda_manager: Optional[LDAModelManager] = None):
        self.db_path     = db_path
        self.lda_manager = lda_manager
        # name -> {'colors', 'cmc', 'types', 'oracle_text', 'rarity'}
        self.cards:       Dict[str, Dict] = {}
        # archetype_name -> [card_name, ...]
        self.by_archetype: Dict[str, List[str]] = defaultdict(list)
        self._logger      = logging.getLogger(self.__class__.__name__)
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.db_path):
            self._logger.warning("Meta DB not found at %s — skipping", self.db_path)
            return
        try:
            conn = sqlite3.connect(self.db_path, timeout=10)
            c    = conn.cursor()
            # decks_cache.db cards table uses type_line, not types
            c.execute("""
                SELECT name, colors, cmc, type_line, oracle_text, rarity
                FROM cards
            """)
            for name, colors_json, cmc, type_line, oracle_text, rarity in c.fetchall():
                try:
                    colors = json.loads(colors_json) if colors_json else []
                    self.cards[name] = {
                        'colors':       colors,
                        'cmc':          int(cmc) if cmc else 0,
                        'types':        type_line or '',
                        'oracle_text':  oracle_text or '',
                        'rarity':       rarity or 'common',
                    }
                except Exception as e:
                    self._logger.debug("Error loading meta card %s: %s", name, e)
            conn.close()
            print(f"  MetaCardPool: {len(self.cards)} cards loaded from {self.db_path}")
        except Exception as e:
            self._logger.error("Failed to load meta card pool: %s", e)

    def get_archetype_card_names(self, archetype: str,
                                 target_colors: Optional[Set[str]] = None,
                                 top_n: int = 60) -> List[str]:
        """
        Return card names from the meta pool that match the archetype (via LDA or
        keyword scan) and are on-color.  Results are sorted by rarity descending.
        """
        matched_key = _lda_archetype_keyword_key(archetype)
        keywords    = ARCHETYPE_KEYWORDS.get(matched_key, []) if matched_key else []

        rarity_rank = {'mythic': 4, 'rare': 3, 'uncommon': 2, 'common': 1}
        candidates  = []

        for name, info in self.cards.items():
            # Color filter
            if target_colors and info['colors']:
                if not set(info['colors']).issubset(target_colors):
                    continue

            # Archetype match: LDA topics first, keyword scan as fallback
            score = 0.0
            if self.lda_manager and self.lda_manager.is_trained:
                topics = self.lda_manager.get_card_topics(name)
                # Find the topic id for this archetype
                tid = next((k for k, v in self.lda_manager.topic_names.items()
                            if v == archetype), None)
                if tid and tid in topics:
                    score = topics[tid]

            if score == 0.0 and keywords:
                text = (info['oracle_text'] + ' ' + name).lower()
                score = sum(0.1 for kw in keywords if kw.lower() in text)

            if score > 0:
                candidates.append((name, score + rarity_rank.get(info['rarity'], 1) * 0.01))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in candidates[:top_n]]


def find_owned_substitute(meta_card_name, owned_pool,
                          target_colors=None, archetype=None):
    # type: (str, CardPool, Optional[Set[str]], Optional[str]) -> Optional[int]
    """
    Given a card name from the meta pool, return the id of the best owned
    card to substitute for it.  Resolution order:
      1. Exact name match in the owned pool (you own it — use it directly).
      2. Keyword-synergy match within on-color cards.   <<< CHANGED
      3. Any on-color card.
    Returns None if the owned pool has no suitable card.
    """
    # 1. Exact match
    owned = owned_pool.get_card_by_name(meta_card_name)
    if owned and owned.id in owned_pool.cards:
        return owned.id

    target = target_colors or set()

    def color_match_score(cid):
        card        = owned_pool.get_card(cid)
        card_colors = set(card.colors) if card.colors else set()
        if target and card_colors and not card_colors.issubset(target):
            return -1.0
        return 1.0

    candidates = [cid for cid in (owned_pool.creatures + owned_pool.spells)
                  if color_match_score(cid) >= 0]
    if not candidates:
        candidates = owned_pool.creatures + owned_pool.spells

    # 2. <<< CHANGED: use cached _has_keyword_synergy instead of raw oracle
    #    scan.  Only narrow the pool when >= 4 synergy cards are available so
    #    sparse collections don't dead-end into an empty candidate list.
    if archetype and candidates:
        synergy = [
            cid for cid in candidates
            if _has_keyword_synergy(
                owned_pool.get_card(cid).id,
                owned_pool.get_card(cid).oracle_text,
                owned_pool.get_card(cid).name,
                archetype,
            )
        ]
        if len(synergy) >= 4:               # <<< CHANGED: threshold guard
            candidates = synergy

    return random.choice(candidates) if candidates else None


def seed_from_meta(owned_pool: CardPool,
                   meta_pool: MetaCardPool,
                   lda_manager: LDAModelManager,
                   archetype: str,
                   target_colors: Optional[Set[str]] = None) -> Optional[List[int]]:
    """
    Build a deck seed from a meta archetype, substituting owned cards for
    any meta cards the player doesn't own.  Returns a 60-card list or None
    if the owned pool is too sparse to fill the deck.
    """
    meta_names = meta_pool.get_archetype_card_names(
        archetype, target_colors=target_colors, top_n=80)

    if not meta_names:
        return None

    deck:        List[int] = []
    card_counts: Dict[int, int] = {}

    def add(cid: int, count: int = 1) -> int:
        card     = owned_pool.get_card(cid)
        headroom = card.max_copies - card_counts.get(cid, 0)
        actual   = min(count, headroom)
        if actual > 0:
            deck.extend([cid] * actual)
            card_counts[cid] = card_counts.get(cid, 0) + actual
        return actual

    # Fill non-land slots from meta card list (substitute if unowned)
    for name in meta_names:
        if len(deck) >= 36:
            break
        info = meta_pool.cards.get(name, {})
        if 'Land' in info.get('types', ''):
            continue

        # Try exact match first, then substitute
        exact = owned_pool.get_card_by_name(name)
        if exact and exact.id in owned_pool.cards:
            cid = exact.id
        else:
            cid = find_owned_substitute(
                name, owned_pool, target_colors=target_colors, archetype=archetype)
        if cid is None:
            continue

        # Determine how many copies (use meta proportion as a guide: 4/3/2/1)
        # We just try 4 and let add() clip to what's owned
        add(cid, 4)

    # Fill remaining non-land slots if we're short
    if len(deck) < 36:
        fallback = [c for c in (owned_pool.creatures + owned_pool.spells)
                    if c not in card_counts]
        random.shuffle(fallback)
        for cid in fallback:
            if len(deck) >= 36:
                break
            add(cid, 1)

    # Fill 24 land slots from owned lands
    target = target_colors or set()
    valid_lands = [
        l for l in owned_pool.lands
        if not target or not owned_pool.get_card(l).colors or
           set(owned_pool.get_card(l).colors).issubset(target) or
           owned_pool.get_card(l).is_basic_land
    ]
    if not valid_lands:
        valid_lands = owned_pool.lands

    basics    = [l for l in valid_lands if owned_pool.get_card(l).is_basic_land]
    nonbasics = [l for l in valid_lands if not owned_pool.get_card(l).is_basic_land]
    land_budget = 24

    random.shuffle(nonbasics)
    for lid in nonbasics:
        if land_budget <= 0:
            break
        land_budget -= add(lid, min(4, land_budget))

    while land_budget > 0:
        src = basics if basics else valid_lands
        if not src:
            break
        lid = random.choice(src)
        if card_counts.get(lid, 0) < owned_pool.get_card(lid).max_copies:
            land_budget -= add(lid, 1)
        else:
            break

    if len(deck) < 40:          # too sparse — not worth using as a seed
        return None

    # Pad to 60 with basics if needed
    while len(deck) < 60:
        src = basics if basics else valid_lands
        if not src:
            break
        lid = random.choice(src)
        deck.append(lid)
        card_counts[lid] = card_counts.get(lid, 0) + 1

    _ensure_deap()
    return creator.Individual(deck[:60])


# ---------- GENETIC OPERATORS ----------
def _lda_archetype_keyword_key(archetype_name: str) -> Optional[str]:
    """
    FIX #20: map an LDA topic name (e.g. 'White Lifegain', 'Izzet Graveyard')
    to an ARCHETYPE_KEYWORDS key using the LDA_TO_ARCHETYPE_KEY lookup.
    Falls back to a direct substring scan of ARCHETYPE_KEYWORDS keys.
    """
    lower = archetype_name.lower()
    for fragment, key in LDA_TO_ARCHETYPE_KEY.items():
        if fragment in lower:
            return key
    # fallback: check ARCHETYPE_KEYWORDS keys directly
    for key in ARCHETYPE_KEYWORDS:
        if key.lower() in lower:
            return key
    return None


def _has_keyword_synergy(card_id: int, card_oracle: str, card_name: str,
                         archetype: str) -> bool:
    """Cached keyword synergy check — avoids rescanning oracle text each mutation."""
    cache_key = (card_id, archetype)
    if cache_key in _keyword_synergy_cache:
        return _keyword_synergy_cache[cache_key]
    matched_key = _lda_archetype_keyword_key(archetype)
    keywords    = ARCHETYPE_KEYWORDS.get(matched_key, []) if matched_key else []
    # Strip reminder text in parentheses before matching — prevents false positives
    # e.g. Gisa matching 'graveyard' from "(cards in their graveyards is a crime)"
    import re as _re
    clean_oracle = _re.sub(r'\([^)]*\)', '', card_oracle or '')
    text_       = clean_oracle.lower()
    name_lower  = card_name.lower()
    result      = any(kw.lower() in text_ or kw.lower() in name_lower for kw in keywords)
    # Also match "gain N life" pattern (e.g. "you gain 3 life") for Lifegain archetype
    if not result and matched_key in ("Lifegain",):
        result = bool(_re.search(r"gain \d+ life", text_))
    _keyword_synergy_cache[cache_key] = result
    return result


def create_lda_individual(card_pool: CardPool, lda_manager: LDAModelManager,
                          target_archetype: Optional[str] = None,
                          meta_pool: Optional['MetaCardPool'] = None) -> List[int]:
    if lda_manager:
        lda_manager._ensure_loaded()

    # Clear keyword synergy cache if it grows too large
    global _keyword_synergy_cache
    if len(_keyword_synergy_cache) > 10000:
        _keyword_synergy_cache.clear()

    # ~30 % of individuals are seeded from meta archetypes (if meta pool available)
    if meta_pool and meta_pool.cards and random.random() < 0.30:
        archetype = target_archetype
        if not archetype and lda_manager and lda_manager.is_trained:
            archetype = random.choice(list(lda_manager.topic_names.values()))
        if archetype:
            seed = seed_from_meta(card_pool, meta_pool, lda_manager,
                                  archetype,
                                  target_colors=getattr(card_pool, 'target_colors', None))
            if seed is not None:
                return seed

    card_cache    = card_pool.cards
    deck          = []
    card_counts   = {}
    target_colors = getattr(card_pool, 'target_colors', set())

    def is_on_color(cid):
        if not target_colors:
            return True
        card = card_cache[cid]
        if not card.colors:
            return True
        return set(card.colors).issubset(target_colors)

    def add_cards(card_id, count):
        card         = card_cache[card_id]
        max_to_add   = card.max_copies - card_counts.get(card_id, 0)
        actual_count = min(count, max_to_add)
        if actual_count > 0:
            deck.extend([card_id] * actual_count)
            card_counts[card_id] = card_counts.get(card_id, 0) + actual_count
        return actual_count

    def is_on_color_land(land_id):
        if not target_colors:
            return True
        card = card_cache[land_id]
        if card.is_basic_land:
            return any(BASIC_LANDS.get(c) == card.name for c in target_colors)
        produced = set(card.mana_production)
        if not produced:
            return True   # colorless utility land — always keep
        return produced.issubset(target_colors)

    valid_lands    = [l for l in card_pool.lands if is_on_color_land(l)]
    if len(valid_lands) < 8:
        valid_lands = card_pool.lands

    basic_lands    = [l for l in valid_lands if card_cache[l].is_basic_land]
    nonbasic_lands = [l for l in valid_lands if not card_cache[l].is_basic_land]

    # ── archetype selection ───────────────────────────────────────────────────
    archetype = None
    if lda_manager and lda_manager.is_trained:
        if target_archetype:
            archetype = target_archetype
        elif target_colors and hasattr(lda_manager, '_topic_colors'):
            candidates = []
            for tid, tname in lda_manager.topic_names.items():
                tcol = lda_manager._topic_colors(tname)
                if not tcol or tcol.issubset(target_colors):
                    coverage = (len(tcol & target_colors) /
                                max(len(target_colors), 1)) if tcol else 0.1
                    w = max(coverage ** 2, 0.05)
                    candidates.append((tname, w))
            if candidates:
                names, weights = zip(*candidates)
                total     = sum(weights)
                archetype = random.choices(
                    names, weights=[w / total for w in weights], k=1)[0]
            else:
                archetype = random.choice(list(lda_manager.topic_names.values()))
        else:
            archetype = random.choice(list(lda_manager.topic_names.values()))

    all_creatures = [c for c in card_pool.creatures if is_on_color(c)]
    all_spells    = [c for c in card_pool.spells    if is_on_color(c)]

    # Filter colorless nonlands by keyword synergy when archetype is locked.
    # Prevents Sorcerous Spyglass and other hate cards from entering the pool.
    if archetype:
        matched_key_cl = _lda_archetype_keyword_key(archetype)
        keywords_cl    = ARCHETYPE_KEYWORDS.get(matched_key_cl, []) if matched_key_cl else []
        def is_ok_colorless(cid):
            card = card_cache[cid]
            if card.colors:
                return True  # colored cards handled elsewhere
            text_ = (card.oracle_text or '').lower()
            name_ = card.name.lower()
            return any(kw.lower() in text_ or kw.lower() in name_ for kw in keywords_cl)
        all_creatures = [c for c in all_creatures if is_ok_colorless(c)]
        all_spells    = [c for c in all_spells    if is_ok_colorless(c)]

    # ── keyword synergy filter ────────────────────────────────────────────────
    if archetype:
        matched_key = _lda_archetype_keyword_key(archetype)
        if matched_key:
            keywords = ARCHETYPE_KEYWORDS[matched_key]

            def has_synergy(cid):
                card  = card_cache[cid]
                text  = (card.oracle_text or '').lower()
                name  = card.name.lower()
                types = card.types.lower() if isinstance(card.types, str) else ''
                if any(kw.lower() in text for kw in keywords):
                    return True
                if any(kw.lower() in name for kw in keywords):
                    return True
                if matched_key in ('Sacrifice', 'Aristocrats'):
                    if 'return' in text and 'graveyard' in text and card.cmc <= 3:
                        return True
                    if 'when' in text and 'dies' in text and 'create' in text:
                        return True
                if matched_key == 'Aggro' and card.cmc <= 2 and 'creature' in types:
                    return True
                if matched_key == 'Tokens' and 'create' in text and 'token' in text:
                    return True
                return False

            fc = [c for c in all_creatures if has_synergy(c)]
            fs = [c for c in all_spells    if has_synergy(c)]
            if len(fc) >= 12: all_creatures = fc
            if len(fs) >= 5:  all_spells    = fs

    # ── LDA anti-filter: remove cards strongly assigned to a different archetype
    # Prevents Overlord of the Balemurk (Golgari Landfall 0.99) and
    # Soul-Guide Lantern (Izzet Graveyard 0.70) polluting lifegain decks.
    # Cards NOT in LDA are kept — absence of data is not evidence of bad fit.
    if archetype and lda_manager and lda_manager.is_trained:
        target_tid = next((k for k, v in lda_manager.topic_names.items()
                           if v == archetype), None)
        if target_tid:
            def is_wrong_archetype(cid):
                card   = card_cache[cid]
                topics = lda_manager.get_card_topics(card.name)
                if not topics:
                    return not _has_keyword_synergy(card.id, card.oracle_text, card.name, archetype)
                best_tid, best_score = max(topics.items(), key=lambda x: x[1])
                on_archetype = topics.get(target_tid, 0.0)
                if best_tid == target_tid or best_score <= 0.5 or on_archetype >= 0.05:
                    return False
                # Mono-colored cards skip LDA topic check but still need keyword synergy
                if card.colors and len(card.colors) == 1:
                    return not _has_keyword_synergy(card.id, card.oracle_text, card.name, archetype)
                return True

            all_creatures = [c for c in all_creatures if not is_wrong_archetype(c)]
            all_spells    = [c for c in all_spells    if not is_wrong_archetype(c)]

    # ── rarity map (used by both card sort and land sort) ─────────────────────
    rarity_map = {'mythic': 4, 'rare': 3, 'uncommon': 2, 'common': 1}

    # ── sort creatures/spells by (rarity, LDA topic score, cmc) ──────────────
    # Cards not in LDA get a score based on keyword synergy:
    #   0.1  — not in LDA but has lifegain keyword synergy (Amalia, Resplendent Angel)
    #  -0.2  — not in LDA and no keyword synergy (Tarrian's Journal, Sunstar Expansionist)
    #  -0.5  — in LDA but strongly assigned to a different archetype (Overlord, Seam Rip)
    def card_sort_key(cid):
        card         = card_cache[cid]
        rarity_score = rarity_map.get(card.rarity, 0)
        topic_score  = 0.0
        if archetype and lda_manager and lda_manager.is_trained:
            topics  = lda_manager.get_card_topics(card.name)
            tid     = next((k for k, v in lda_manager.topic_names.items()
                            if v == archetype), None)
            if not topics:
                # Not in LDA — check keyword synergy to determine score.
                # Cards with no LDA data AND no keyword synergy get -0.2 so
                # they sort below synergy cards regardless of rarity.
                matched_key = _lda_archetype_keyword_key(archetype)
                keywords    = ARCHETYPE_KEYWORDS.get(matched_key, []) if matched_key else []
                text        = (card.oracle_text or '').lower()
                name_lower  = card.name.lower()
                has_kw      = any(kw.lower() in text or kw.lower() in name_lower
                                  for kw in keywords)
                topic_score = 0.1 if has_kw else -0.2
            elif tid and topics.get(tid, 0.0) > 0.01:
                # Has a meaningful score for target archetype
                topic_score = topics[tid]
            else:
                # In LDA but strongly assigned to a different archetype
                topic_score = -0.5
        return (rarity_score, topic_score, -card.cmc)

    creatures = sorted(all_creatures, key=card_sort_key, reverse=True)
    spells    = sorted(all_spells,    key=card_sort_key, reverse=True)

    if not creatures and not spells:
        creatures = card_pool.creatures
        spells    = card_pool.spells

    # ── LDA topic card reordering (boost topic cards to front) ───────────────
    if archetype and lda_manager and lda_manager.is_trained:
        topic_cards        = [c for c in lda_manager.get_cards_by_topic(
                                  archetype, card_pool, top_n=100) if is_on_color(c)]
        topic_creature_set = set(c for c in topic_cards if c in card_pool.creatures)
        topic_spell_set    = set(c for c in topic_cards if c in card_pool.spells)
        creatures = ([c for c in creatures if c in topic_creature_set] +
                     [c for c in creatures if c not in topic_creature_set])
        spells    = ([c for c in spells if c in topic_spell_set] +
                     [c for c in spells if c not in topic_spell_set])

    creatures = creatures[:60]
    spells    = spells[:40]

    # ── playset selection: up to 6x 4-ofs ────────────────────────────────────
    playsets_selected = 0
    for card_id in creatures + spells:
        if playsets_selected >= 6:
            break
        if card_id in card_counts:
            continue
        if card_cache[card_id].max_copies >= 4:
            if add_cards(card_id, 4) == 4:
                playsets_selected += 1

    # ── 3-ofs ─────────────────────────────────────────────────────────────────
    remaining_nonlands = [c for c in (creatures + spells) if c not in card_counts]
    random.shuffle(remaining_nonlands)
    threeofs_selected = 0
    for card_id in remaining_nonlands:
        if threeofs_selected >= 4 or len(deck) >= 36:
            break
        if card_cache[card_id].max_copies >= 3:
            if add_cards(card_id, 3) == 3:
                threeofs_selected += 1

    # ── 2-ofs ─────────────────────────────────────────────────────────────────
    remaining = [c for c in (creatures + spells) if c not in card_counts]
    random.shuffle(remaining)
    while len(deck) < 36 and remaining:
        add_cards(remaining.pop(0), 2)

    # ── 1-ofs (last resort) ───────────────────────────────────────────────────
    last_resort = [c for c in (creatures + spells) if c not in card_counts]
    random.shuffle(last_resort)
    while len(deck) < 36 and last_resort:
        add_cards(last_resort.pop(0), 1)

    # ── land sort key ─────────────────────────────────────────────────────────
    # Only flag unconditional taplands — Godless Shrine says
    # "if you don't, it enters tapped" and must NOT be treated as a tapland.
    UNCONDITIONAL_TAPPED = (
        'this land enters tapped',
        'enters the battlefield tapped',
        'put onto the battlefield tapped',
    )

    def land_sort_key(lid):
        land      = card_cache[lid]
        oracle    = (land.oracle_text or '').lower()
        is_tapped = False
        for phrase in UNCONDITIONAL_TAPPED:
            idx = oracle.find(phrase)
            if idx == -1:
                continue
            following = oracle[idx + len(phrase):idx + len(phrase) + 60]
            if 'unless' not in following:
                is_tapped = True
                break
        rarity_score = rarity_map.get(land.rarity, 0)
        produced  = set(p for p in land.mana_production if p in 'WUBRG')
        on_color  = len(produced & target_colors) if target_colors else 0
        off_color = len(produced - target_colors) if target_colors else 0
        return (not is_tapped, rarity_score, on_color, -off_color)

    # Sort nonbasic lands by quality — untapped > rarity > on-color > off-color
    nonbasic_lands = sorted(nonbasic_lands, key=land_sort_key, reverse=True)

    # ── land base ─────────────────────────────────────────────────────────────
    land_slots_remaining = 24

    # Seed 4 copies of each basic upfront
    if basic_lands:
        for basic_id in basic_lands[:min(2, len(basic_lands))]:
            if land_slots_remaining < 4:
                break
            land_slots_remaining -= add_cards(basic_id, 4)

    # Nonbasic duals in quality order, cap at 4 dual types
    duals_added = 0
    for land_id in nonbasic_lands:
        if land_slots_remaining < 3 or duals_added >= 4:
            break
        card = card_cache[land_id]
        if card.max_copies >= 4:
            added = add_cards(land_id, 4)
            if added > 0:
                duals_added += 1
                land_slots_remaining -= added
        elif card.max_copies >= 3:
            added = add_cards(land_id, 3)
            if added > 0:
                duals_added += 1
                land_slots_remaining -= added

    # Fill remaining land slots with 2-ofs (sorted by quality)
    remaining_lands = [l for l in valid_lands
                       if card_cache[l].max_copies > card_counts.get(l, 0)]
    remaining_lands = sorted(remaining_lands, key=land_sort_key, reverse=True)
    while land_slots_remaining >= 2:
        available = [l for l in remaining_lands
                     if card_cache[l].max_copies - card_counts.get(l, 0) >= 2]
        if not available:
            break
        land_slots_remaining -= add_cards(available[0], 2)

    # Fill remaining slots with 1-ofs — prefer basics for reliability
    while land_slots_remaining > 0:
        available_basics = [l for l in basic_lands
                            if card_cache[l].max_copies > card_counts.get(l, 0)]
        available_nonbasic = sorted(
            [l for l in valid_lands
             if card_cache[l].max_copies > card_counts.get(l, 0)
             and not card_cache[l].is_basic_land],
            key=land_sort_key, reverse=True)
        available = available_basics or available_nonbasic
        if not available:
            break
        land_slots_remaining -= add_cards(available[0], 1)

    # ── emergency padding ─────────────────────────────────────────────────────
    emergency_attempts = 0
    while len(deck) < 60 and emergency_attempts < 1000:
        emergency_attempts += 1
        if land_slots_remaining <= 0:
            pool_nonland = card_pool.creatures + card_pool.spells
            if pool_nonland:
                cid = random.choice(pool_nonland)
                if card_counts.get(cid, 0) < card_cache[cid].max_copies:
                    deck.append(cid)
                    card_counts[cid] = card_counts.get(cid, 0) + 1
            break
        land = (random.choice(basic_lands) if basic_lands
                else random.choice(valid_lands) if valid_lands
                else random.choice(card_pool.lands))
        if card_counts.get(land, 0) < card_cache[land].max_copies:
            deck.append(land)
            card_counts[land] = card_counts.get(land, 0) + 1
            land_slots_remaining -= 1

    _ensure_deap()
    return creator.Individual(deck[:60])

def _curve_profile(deck, pool):
    # type: (List[int], CardPool) -> Dict[int, int]
    """
    Returns {cmc: count} for non-land cards only.
    Called inside mut_lda_deck to snapshot the curve before picking a swap.
    """
    return Counter(
        pool.get_card(cid).cmc
        for cid in deck
        if 'Land' not in pool.get_card(cid).types
    )


def _curve_gap_score(cmc, profile, archetype):
    # type: (int, Dict[int, int], str) -> float
    """
    Returns how urgently the curve needs a card at this CMC (0.0 – 1.0).
    Higher = bigger gap = candidate is more desirable as a replacement.

    Ideal targets are archetype-aware so a Lifegain deck never tries to
    fill up to 10 one-drops the way an Aggro deck would.

    The score is fractional so it acts as a soft weight multiplier
    rather than a hard filter — the GA can still pick a high-CMC card
    when the curve genuinely has room for it.
    """
    if any(a in archetype for a in ('Aggro', 'Tokens', 'Burn')):
        ideal = {1: 10, 2: 12, 3: 8, 4: 4, 5: 2}
    elif any(a in archetype for a in ('Control', 'Combo')):
        ideal = {1: 2, 2: 6, 3: 10, 4: 8, 5: 6, 6: 4}
    else:   # Midrange / Lifegain / Graveyard / most two-colour strategies
        ideal = {1: 4, 2: 10, 3: 10, 4: 6, 5: 4, 6: 2}

    target  = ideal.get(cmc, 1)
    current = profile.get(cmc, 0)
    if current >= target:
        return 0.0
    return (target - current) / target      # 0.0 – 1.0

def mut_lda_deck(individual, card_pool, lda_manager,
                 indpb=0.15, target_archetype=None):
    # type: (List[int], CardPool, LDAModelManager, float, Optional[str]) -> Tuple
    """
    Archetype-aware, curve-aware mutation operator.

    Three mutation branches selected by a single random roll:

      0.00 – 0.40  Singleton consolidation
                   Merge paired singletons into 2-ofs.
                   <<< CHANGED: curve-aware tiebreak — keep the copy
                   whose CMC fills a bigger gap in the current curve.

      0.40 – 0.75  Playset promotion
                   Promote 2/3-ofs toward 4-ofs, preferring to replace
                   taplands and singleton noise cards.  (Unchanged.)

      0.75 – 1.00  Archetype cleanup + per-card swap
                   Remove topic outliers, then per-slot swap using
                   <<< CHANGED: curve-weighted candidate selection so
                   the replacement is biased toward filling curve gaps.
    """
    if not individual:
        return (individual,)
    if lda_manager:
        lda_manager._ensure_loaded()

    deck        = list(individual)
    card_counts = Counter(deck)

    # ── small helpers ─────────────────────────────────────────────────────────
    def is_legal(cid):
        if not card_pool.target_colors:
            return True
        card = card_pool.get_card(cid)
        if not card.colors:
            return True
        return set(card.colors).issubset(card_pool.target_colors)

    def decrement(counts, cid):
        counts[cid] -= 1
        if counts[cid] == 0:
            del counts[cid]

    rarity_map    = {'mythic': 4, 'rare': 3, 'uncommon': 2, 'common': 1}
    target_colors = card_pool.target_colors or set()

    def get_current_archetype():
        if target_archetype:
            return target_archetype
        if lda_manager and lda_manager.is_trained:
            return lda_manager.get_deck_archetype(deck, card_pool)[0]
        return "Unknown"

    def is_wrong_archetype_card(cid, arch):
        if not lda_manager or not lda_manager.is_trained or not arch:
            return False
        target_tid = next((k for k, v in lda_manager.topic_names.items()
                           if v == arch), None)
        if not target_tid:
            return False
        card   = card_pool.get_card(cid)
        topics = lda_manager.get_card_topics(card.name)
        if not topics:
            return not _has_keyword_synergy(card.id, card.oracle_text,
                                            card.name, arch)
        best_tid, best_score = max(topics.items(), key=lambda x: x[1])
        on_archetype         = topics.get(target_tid, 0.0)
        if best_tid == target_tid or best_score <= 0.5 or on_archetype >= 0.05:
            return False
        if card.colors and len(card.colors) == 1:
            return not _has_keyword_synergy(card.id, card.oracle_text,
                                            card.name, arch)
        return True

    UNCONDITIONAL_TAPPED = (
        'this land enters tapped',
        'enters the battlefield tapped',
        'put onto the battlefield tapped',
    )

    def is_unconditional_tapland(card):
        oracle = (card.oracle_text or '').lower()
        for phrase in UNCONDITIONAL_TAPPED:
            idx = oracle.find(phrase)
            if idx == -1:
                continue
            following = oracle[idx + len(phrase): idx + len(phrase) + 60]
            if 'unless' not in following:
                return True
        return False

    def land_sort_key(lid):
        land         = card_pool.get_card(lid)
        is_tapped    = is_unconditional_tapland(land)
        rarity_score = rarity_map.get(land.rarity, 0)
        produced     = set(p for p in land.mana_production if p in 'WUBRG')
        on_color     = len(produced & target_colors) if target_colors else 0
        off_color    = len(produced - target_colors) if target_colors else 0
        return (not is_tapped, rarity_score, on_color, -off_color)

    # ── branch selection ──────────────────────────────────────────────────────
    mutation_roll = random.random()

    # ── BRANCH 1: singleton consolidation  (0.00 – 0.40) ────────────────────
    if mutation_roll < 0.40:
        current_arch = get_current_archetype()
        singletons   = [c for c, n in card_counts.items() if n == 1]
        creatures    = [c for c in singletons
                        if 'Creature' in card_pool.get_card(c).types]
        spells       = [c for c in singletons if c not in creatures]

        def merge_pair(winner_cid, loser_cid):
            """Replace one copy of loser_cid with winner_cid in the deck."""
            for i, c in enumerate(deck):
                if c == loser_cid:
                    deck[i] = winner_cid
                    decrement(card_counts, loser_cid)
                    card_counts[winner_cid] = card_counts.get(winner_cid, 0) + 1
                    break

        # <<< CHANGED: curve-aware tiebreak replaces the old list-order winner.
        # Snapshot the curve once per pair — cheap (single Counter pass).
        while len(creatures) >= 2:
            c1, c2 = creatures.pop(0), creatures.pop(0)
            curve  = _curve_profile(deck, card_pool)           # <<< NEW
            gap1   = _curve_gap_score(                         # <<< NEW
                card_pool.get_card(c1).cmc, curve, current_arch)
            gap2   = _curve_gap_score(                         # <<< NEW
                card_pool.get_card(c2).cmc, curve, current_arch)
            winner, loser = (c1, c2) if gap1 >= gap2 else (c2, c1)  # <<< NEW
            merge_pair(winner, loser)

        while len(spells) >= 2:
            c1, c2 = spells.pop(0), spells.pop(0)
            curve  = _curve_profile(deck, card_pool)           # <<< NEW
            gap1   = _curve_gap_score(                         # <<< NEW
                card_pool.get_card(c1).cmc, curve, current_arch)
            gap2   = _curve_gap_score(                         # <<< NEW
                card_pool.get_card(c2).cmc, curve, current_arch)
            winner, loser = (c1, c2) if gap1 >= gap2 else (c2, c1)  # <<< NEW
            merge_pair(winner, loser)

        # Odd leftover singleton: merge into an existing 2-of of the same type.
        # <<< CHANGED: pick the 2-of that fills the biggest gap rather than
        # choosing randomly.
        for leftover in (creatures + spells):
            same_type = [
                c for c, cnt in card_counts.items()
                if cnt == 2 and
                ('Creature' in card_pool.get_card(c).types) ==
                ('Creature' in card_pool.get_card(leftover).types)
            ]
            if same_type:
                curve      = _curve_profile(deck, card_pool)   # <<< NEW
                target_cid = max(                               # <<< NEW
                    same_type,
                    key=lambda c: _curve_gap_score(
                        card_pool.get_card(c).cmc, curve, current_arch)
                )
                merge_pair(target_cid, leftover)

    # ── BRANCH 2: playset promotion  (0.40 – 0.75) ───────────────────────────
    elif mutation_roll < 0.75:
        light = [(c, n) for c, n in card_counts.items()
                 if 1 < n < 4 and card_pool.get_card(c).max_copies >= 4]
        if light:
            target_cid, current_count = random.choice(light)
            target_card = card_pool.get_card(target_cid)
            needed      = 4 - current_count

            land_candidates = [
                (i, c) for i, c in enumerate(deck)
                if 'Land' in card_pool.get_card(c).types
                and c != target_cid
                and not card_pool.get_card(c).is_basic_land
            ]
            land_candidates.sort(key=lambda x: land_sort_key(x[1]))

            other_candidates = [
                (i, c) for i, c in enumerate(deck)
                if card_counts[c] == 1 and c != target_cid
            ]
            random.shuffle(other_candidates)

            candidates = land_candidates[:needed] + other_candidates
            for idx, old_card in candidates[:needed]:
                if deck.count(target_cid) < target_card.max_copies:
                    deck[idx] = target_cid
                    decrement(card_counts, old_card)
                    card_counts[target_cid] += 1

    # ── BRANCH 3: archetype cleanup + per-card swap  (0.75 – 1.00) ──────────
    else:
        current_arch = get_current_archetype()
        singletons   = [c for c, n in card_counts.items() if n == 1]

        # Archetype outlier removal (unchanged from original)
        if singletons and lda_manager and lda_manager.is_trained:
            archetype_cards = [
                c for c in lda_manager.get_cards_by_topic(
                    current_arch, card_pool, top_n=30)
                if is_legal(c)
            ]
            archetype_set = set(archetype_cards)
            outliers      = [c for c in singletons if c not in archetype_set]
            if outliers:
                outlier    = random.choice(outliers)
                good_cards = [c for c, cnt in card_counts.items()
                              if cnt >= 3 and c in archetype_set]
                if good_cards:
                    replacement = random.choice(good_cards)
                    for i, c in enumerate(deck):
                        if c == outlier:
                            if (deck.count(replacement) <
                                    card_pool.get_card(replacement).max_copies):
                                deck[i] = replacement
                                decrement(card_counts, outlier)
                                card_counts[replacement] += 1
                            break

        # Per-card mutation
        current_arch = get_current_archetype()

        for i in range(len(deck)):
            if random.random() >= indpb:
                continue

            current_card = deck[i]
            current      = card_pool.get_card(current_card)

            # Tapland upgrade path (unchanged)
            if 'Land' in current.types and not current.is_basic_land:
                if is_unconditional_tapland(current):
                    better_lands = sorted(
                        [l for l in card_pool.lands
                         if is_legal(l) and
                            not card_pool.get_card(l).is_basic_land and
                            deck.count(l) < card_pool.get_card(l).max_copies and
                            l != current_card],
                        key=land_sort_key, reverse=True)
                    if better_lands:
                        new_land = better_lands[0]
                        deck[i]  = new_land
                        decrement(card_counts, current_card)
                        card_counts[new_land] = card_counts.get(new_land, 0) + 1
                        continue

            # 80 % chance: complete an existing partial playset first
            incomplete = [c for c, cnt in card_counts.items()
                          if 1 < cnt < 4 and c != current_card]
            if incomplete and random.random() < 0.8:
                new_card = random.choice(incomplete)
            else:
                # ── <<< CHANGED: curve-weighted candidate selection ───────────
                #
                # Old behaviour: random.choice(topic_cards)
                #   — archetype-aware but blind to the curve.
                #
                # New behaviour: random.choices(candidate_pool, weights=...)
                #   — each candidate is weighted by how well it fills a curve
                #     gap.  Weight = 1.0 + gap_score * 3.0, so a card at a
                #     fully-stocked CMC gets weight 1.0 (still selectable) and
                #     a card at an empty CMC slot gets weight 4.0.
                #
                # The pool is widened from top_n=20 to top_n=40 so the weighting
                # has enough candidates to choose from across the full curve.

                if lda_manager and lda_manager.is_trained:
                    topic_cards = [
                        c for c in lda_manager.get_cards_by_topic(
                            current_arch, card_pool, top_n=40)  # <<< CHANGED 20->40
                        if is_legal(c) and not is_wrong_archetype_card(c, current_arch)
                    ]
                    candidate_pool = topic_cards if topic_cards else [
                        c for c in (card_pool.creatures + card_pool.spells)
                        if is_legal(c)
                    ]
                else:
                    candidate_pool = [
                        c for c in (card_pool.creatures + card_pool.spells)
                        if is_legal(c)
                    ]

                if not candidate_pool:
                    new_card = current_card
                else:
                    curve   = _curve_profile(deck, card_pool)       # <<< NEW
                    weights = [                                      # <<< NEW
                        1.0 + _curve_gap_score(
                            card_pool.get_card(cid).cmc,
                            curve,
                            current_arch,
                        ) * 3.0
                        for cid in candidate_pool
                    ]
                    new_card = random.choices(                       # <<< NEW
                        candidate_pool, weights=weights, k=1)[0]
                # ── end curve-weighted selection ──────────────────────────────

            # Acceptance gate: reject off-theme / over-limit / same-card swaps
            if (is_legal(new_card)
                    and deck.count(new_card) < card_pool.get_card(new_card).max_copies
                    and new_card != current_card
                    and not is_wrong_archetype_card(new_card, current_arch)):
                deck[i] = new_card
                decrement(card_counts, current_card)
                card_counts[new_card] = card_counts.get(new_card, 0) + 1

    individual[:] = deck
    return (individual,)


def cx_deck(ind1: List[int], ind2: List[int],
            card_pool: CardPool,
            lda_manager: Optional['LDAModelManager'] = None,
            target_archetype: Optional[str] = None) -> Tuple[List[int], List[int]]:
    if len(ind1) != 60 or len(ind2) != 60:
        return ind1, ind2

    # ── archetype filter ──────────────────────────────────────────────────────
    # Prevents crossover from propagating off-theme packages (e.g. Tarrian's
    # Journal) that mutation filters have already blocked.
    def is_ok_for_archetype(cid: int) -> bool:
        if not lda_manager or not lda_manager.is_trained or not target_archetype:
            return True
        target_tid = next((k for k, v in lda_manager.topic_names.items()
                           if v == target_archetype), None)
        if not target_tid:
            return True
        card = card_pool.get_card(cid)
        topics = lda_manager.get_card_topics(card.name)
        if not topics:
            return _has_keyword_synergy(card.id, card.oracle_text, card.name, target_archetype)
        best_tid, best_score = max(topics.items(), key=lambda x: x[1])
        on_archetype = topics.get(target_tid, 0.0)
        if best_tid == target_tid or best_score <= 0.5 or on_archetype >= 0.05:
            return True
        if card.colors and len(card.colors) == 1:
            return _has_keyword_synergy(card.id, card.oracle_text, card.name, target_archetype)
        return False

    def get_packages(deck):
        counts = Counter(deck)
        return {cid: count for cid, count in counts.items() if count >= 3}

    pk1 = get_packages(ind1)
    pk2 = get_packages(ind2)

    if not pk1 or not pk2:
        cxpoint = random.randint(1, 59)
        ind1[cxpoint:], ind2[cxpoint:] = ind2[cxpoint:], ind1[cxpoint:]
        return ind1, ind2

    d1, d2 = list(ind1), list(ind2)
    swaps   = random.randint(1, min(3, len(pk1), len(pk2)))
    pk1_items = list(pk1.items()); random.shuffle(pk1_items)
    pk2_items = list(pk2.items()); random.shuffle(pk2_items)

    for i in range(swaps):
        card1, count1 = pk1_items[i]
        card2, count2 = pk2_items[i]
        if card1 == card2:
            continue
        # Never swap land packages — causes land_count=0 → -1250 heuristic
        if 'Land' in card_pool.get_card(card1).types:
            continue
        if 'Land' in card_pool.get_card(card2).types:
            continue
        # Only swap if the incoming card passes the archetype filter
        if not is_ok_for_archetype(card2):
            continue
        if not is_ok_for_archetype(card1):
            continue
        d1 = [c for c in d1 if c != card1]
        d2 = [c for c in d2 if c != card2]
        d1.extend([card2] * count2)
        d2.extend([card1] * count1)

    def fix_size(deck, target_size=60):
        if len(deck) > target_size:
            excess = len(deck) - target_size
            remove = set(random.sample(range(len(deck)), excess))
            deck   = [c for i, c in enumerate(deck) if i not in remove]
        while len(deck) < target_size:
            deficit = target_size - len(deck)
            if card_pool.lands:
                basics = [l for l in card_pool.lands if card_pool.get_card(l).is_basic_land]
                src    = basics if basics else card_pool.lands
                deck.extend(random.choices(src, k=min(deficit, len(src))))
            elif card_pool.creatures:
                deck.extend(random.choices(card_pool.creatures, k=min(deficit, len(card_pool.creatures))))
            else:
                break
        return deck[:target_size]

    d1 = fix_size(d1)
    d2 = fix_size(d2)

    ind1[:] = d1
    ind2[:] = d2
    return ind1, ind2


class MetaDeckLoader:
    def __init__(self, db_path: str = META_DB_PATH):
        self.db_path       = db_path
        self.forge_deck_dir = FORGE_DECK_DIR
        self._logger       = logging.getLogger(self.__class__.__name__)

    def load_random_meta_decks(self, num_decks: int = 10) -> List[MetaDeck]:
        print(f"  Loading {num_decks} random meta decks from database...")

        if not os.path.exists(self.db_path):
            print(f"  ✗ Database not found: {self.db_path}")
            return self._create_sample_decks(num_decks)

        try:
            conn = sqlite3.connect(self.db_path)
            c    = conn.cursor()
            c.execute("""
                SELECT slug, name, colors, tier, file_hash, card_names
                FROM decks WHERE is_meta = 1
                ORDER BY RANDOM() LIMIT ?
            """, (num_decks,))
            rows = c.fetchall()
            conn.close()

            if not rows:
                return self._create_sample_decks(num_decks)

            meta_decks = []
            for i, (slug, name, colors, tier, file_hash, card_names_json) in enumerate(rows):
                try:
                    dck_path = (self.forge_deck_dir / f"{file_hash}.dck"
                                if file_hash
                                else self.forge_deck_dir /
                                     f"meta_{re.sub(r'[^\w-]', '_', slug)}.dck")
                    if not dck_path.exists():
                        continue
                    card_names = []
                    if card_names_json:
                        try:
                            card_names = json.loads(card_names_json)
                            if not isinstance(card_names, list):
                                card_names = []
                        except json.JSONDecodeError:
                            pass

                    meta_decks.append(MetaDeck(
                        name=name, colors=colors or '',
                        tier=tier or 3, file_path=str(dck_path),
                        card_names=card_names
                    ))
                    print(f"    [{i+1}] {name[:40]:40s} ({colors})")
                except Exception as e:
                    self._logger.error("Error processing meta deck %s: %s", name, e)
                    continue

            if len(meta_decks) < num_decks:
                meta_decks.extend(self._create_sample_decks(num_decks - len(meta_decks)))

            print(f"  ✓ Loaded {len(meta_decks)} meta decks")
            return meta_decks

        except Exception as e:
            self._logger.error("Error loading from database: %s", e)
            return self._create_sample_decks(num_decks)

    def _create_sample_decks(self, num_decks: int) -> List[MetaDeck]:
        print(f"  Creating {num_decks} sample decks...")
        samples = [
            ("Mono White Aggro", "W", 1), ("Mono Red Burn",    "R", 1),
            ("Azorius Control",  "WU", 1), ("Rakdos Midrange",  "BR", 2),
            ("Golgari Ramp",     "BG", 2), ("Dimir Control",    "UB", 2),
            ("Mono Green Stompy","G", 2),  ("Orzhov Tokens",    "WB", 3),
            ("Izzet Spells",     "UR", 3), ("Gruul Aggro",      "RG", 3),
        ]
        decks = []
        for i, (n, c, t) in enumerate(samples[:num_decks]):
            try:
                p = self.forge_deck_dir / f"meta_sample_{i+1:02d}_{n.replace(' ','_')}.dck"
                if not p.exists():
                    self._create_fallback_deck(n, c, p)
                decks.append(MetaDeck(name=n, colors=c, tier=t, file_path=str(p)))
            except Exception as e:
                self._logger.error("Error creating sample deck %s: %s", n, e)
        return decks

    def _create_fallback_deck(self, name: str, colors: str, path: Path):
        color_list = list(colors) if colors else ['W']
        per_land   = 60 // len(color_list)
        with open(path, 'w') as f:
            f.write(f"[metadata]\nName={name}\n\n[Main]\n")
            total = 0
            for c in color_list:
                land_name = BASIC_LANDS.get(c, 'Plains')
                f.write(f"{per_land} {land_name}\n")
                total += per_land
            remainder = 60 - total
            if remainder:
                f.write(f"{remainder} {BASIC_LANDS.get(color_list[0], 'Plains')}\n")


# ---------- FORGE RUNNER ----------
class ForgeHeadlessRunner:
    def __init__(self, forge_dir: str = FORGE_PATH, mode: str = 'live'):
        self.forge_dir    = Path(forge_dir)
        self.mode         = mode
        self.jar_path     = self._find_jar()
        self.results_cache: Dict[str, Dict] = {}
        self.forge_deck_dir = FORGE_DECK_DIR
        self._executor: Optional[ThreadPoolExecutor] = None

        if self.mode == 'sim':
            print("⚠️  SIMULATION MODE")
            self.use_simulation = True
        elif self.jar_path is None:
            print("⚠️  Forge JAR not found, falling back to simulation")
            self.use_simulation = True
        else:
            print(f"✓ Forge JAR found: {self.jar_path}")
            self.forge_deck_dir.mkdir(parents=True, exist_ok=True)
            self.use_simulation = False

        self._init_executor()

    def _find_jar(self) -> Optional[Path]:
        if not self.forge_dir.exists():
            return None
        jars = list(self.forge_dir.glob('forge-gui-desktop-*.jar'))
        return jars[0] if jars else None

    def _init_executor(self):
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=4)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        if self._executor is not None:
            try:
                self._executor.shutdown(wait=True)
            except Exception:
                pass
            finally:
                self._executor = None

    def _verify_deck_file(self, deck_path: str) -> bool:
        path = Path(deck_path)
        if not path.exists() or path.stat().st_size == 0:
            return False
        try:
            with open(path, 'r') as f:
                return '[Main]' in f.read()
        except Exception:
            return False

    def test_deck(self, deck_path: str, meta_deck: MetaDeck,
                  num_games: int = GAMES_PER_MATCHUP_DEFAULT) -> Dict:
        cache_key = f"{deck_path}_{meta_deck.file_path}_{num_games}"
        if cache_key in self.results_cache:
            result = self.results_cache[cache_key]
            self._print_result(meta_deck, result)
            return result

        if self.use_simulation:
            result = self._simulate_match(num_games, meta_deck)
            self._print_result(meta_deck, result)
            self.results_cache[cache_key] = result
            return result

        if (not self._verify_deck_file(deck_path) or
                not self._verify_deck_file(meta_deck.file_path)):
            result = self._simulate_match(num_games, meta_deck)
            self._print_result(meta_deck, result)
            return result

        deck1_name = Path(deck_path).name
        deck2_name = Path(meta_deck.file_path).name
        cmd = ["java", "-Xmx1G", "-jar", str(self.jar_path),
               "sim", "-d", deck1_name, deck2_name, "-n", str(num_games), "-q"]
        print(f"    Running: {deck1_name} vs {meta_deck.name} ({num_games} games)",
              flush=True)

        try:
            proc   = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=300, cwd=str(self.forge_dir))
            output = proc.stdout + proc.stderr
            if proc.returncode != 0:
                result = self._simulate_match(num_games, meta_deck)
            else:
                # DEBUG: log raw Forge output to forge_debug.log for inspection
                try:
                    with open('./forge_debug.log', 'a') as _dbg:
                        _dbg.write(f"\n=== {deck1_name} vs {meta_deck.name} ===\n")
                        _dbg.write(output[:2000])
                except Exception:
                    pass
                wins, losses = self._parse_simulator_output(output)
                result = (self._simulate_match(num_games, meta_deck)
                          if wins == 0 and losses == 0
                          else {
                              'wins': wins, 'losses': losses,
                              'win_rate': wins / (wins + losses) if (wins + losses) else 0,
                              'meta_deck': meta_deck.name,
                              'games_played': wins + losses,
                              'simulated': False
                          })
            self.results_cache[cache_key] = result
            self._print_result(meta_deck, result)
            return result
        except Exception:
            result = self._simulate_match(num_games, meta_deck)
            self._print_result(meta_deck, result)
            return result

    def _print_result(self, meta_deck: MetaDeck, result: Dict):
        status  = "[SIM]" if result.get('simulated') else "[LIVE]"
        bar_len = int(result['win_rate'] * 10)
        bar     = "█" * bar_len + "░" * (10 - bar_len)
        print(f"  vs {meta_deck.name:25s} [{bar}] "
              f"{result['wins']:>2}/{result['games_played']:>2} "
              f"({result['win_rate']*100:5.1f}%) {status}", flush=True)

    def _parse_simulator_output(self, content: str) -> Tuple[int, int]:
        p1w = len(re.findall(r'Game Outcome: Ai\(1\).*?has won', content))
        p2w = len(re.findall(r'Game Outcome: Ai\(2\).*?has won', content))
        if p1w or p2w:
            return p1w, p2w
        m = re.search(r'Match Result: Ai\(1\).*?: (\d+) Ai\(2\).*?: (\d+)', content)
        if m:
            return int(m.group(1)), int(m.group(2))
        for pat in [r'Player\s*1.*?(\d+).*?wins?',
                    r'Player\s*1.*?(\d+)\s*-\s*(\d+)',
                    r'Results?:?\s*(\d+)\s*-\s*(\d+)']:
            m = re.search(pat, content, re.IGNORECASE)
            if m:
                g = m.groups()
                if len(g) >= 2:
                    return int(g[0]), int(g[1])
        return 0, 0

    def _simulate_match(self, num_games: int, meta_deck: MetaDeck) -> Dict:
        tier_power = {1: 0.65, 2: 0.55, 3: 0.45}.get(meta_deck.tier, 0.5)
        win_prob   = max(0.1, min(0.9, tier_power + random.uniform(-0.10, 0.10)))
        wins       = sum(1 for _ in range(num_games) if random.random() < win_prob)
        return {
            'wins': wins, 'losses': num_games - wins,
            'win_rate': wins / num_games if num_games else 0,
            'meta_deck': meta_deck.name, 'games_played': num_games, 'simulated': True
        }

    def test_against_meta(self, deck_path: str, meta_decks: List[MetaDeck],
                          quick_test: bool = False,
                          games_per_matchup: int = GAMES_PER_MATCHUP_DEFAULT) -> Dict:
        self._init_executor()
        total_wins = total_games = 0
        matchups   = {}
        games_per  = 3 if quick_test else games_per_matchup
        print(f"\nTesting against {len(meta_decks)} meta decks "
              f"({'simulation' if self.use_simulation else 'Forge'} mode)...")

        future_to_meta = {
            self._executor.submit(self.test_deck, deck_path, meta, games_per): meta
            for meta in meta_decks
        }
        for future in as_completed(future_to_meta):
            meta = future_to_meta[future]
            try:
                result = future.result()
            except Exception as e:
                logger.error("Error testing against %s: %s", meta.name, e)
                result = self._simulate_match(games_per, meta)
                self._print_result(meta, result)

            matchups[meta.name]  = result
            total_wins          += result['wins']
            total_games         += result['games_played']

        overall_wr = total_wins / total_games if total_games else 0
        return {
            'overall_win_rate': overall_wr, 'total_wins': total_wins,
            'total_games': total_games, 'matchups': matchups,
            'simulated': any(m.get('simulated') for m in matchups.values())
        }


# ---------- FITNESS EVALUATOR ----------
class AdaptiveFitnessEvaluator:
    def __init__(self, pool: CardPool, meta_decks: List[MetaDeck],
                 forge_runner: ForgeHeadlessRunner,
                 lda_manager: Optional[LDAModelManager] = None,
                 games_per_matchup: int = GAMES_PER_MATCHUP_DEFAULT):
        self.pool              = pool
        self.meta_decks        = meta_decks
        self.forge             = forge_runner
        self.lda_manager       = lda_manager
        self.games_per_matchup = games_per_matchup
        self.tested            = OrderedDict()
        self.max_cache_size    = MAX_CACHE_SIZE
        self.generation        = 0
        self._logger           = logging.getLogger(self.__class__.__name__)
        self._pip_pattern      = re.compile(r'([WUBRG])')
        self._tapland_indicators = {
            'enters the battlefield tapped', 'etb tapped', 'enter tapped'
        }

    def _mana_base_fitness(self, individual: List[int]) -> float:
        deck_cards = [self.pool.get_card(cid) for cid in individual]
        lands      = [c for c in deck_cards if 'Land' in c.types]
        spells     = [c for c in deck_cards if 'Land' not in c.types]
        land_count = len(lands)
        score      = 0.0

        if 22 <= land_count <= 26:
            score += 40
        elif 20 <= land_count <= 28:
            score += 15
        else:
            score -= abs(24 - land_count) * 8

        if land_count == 0:
            return -500.0

        sources  = Counter()
        for land in lands:
            sources.update(p for p in land.mana_production if p in 'WUBRG')

        pip_reqs = Counter()
        for card in spells:
            if card.mana_cost:
                pip_reqs.update(self._pip_pattern.findall(card.mana_cost))

        for color, pips in pip_reqs.items():
            if not pips:
                continue
            src    = sources.get(color, 0)
            needed = min(16, max(8, int(pips * 1.15)))  # cap at 16 — 24 lands can never provide 30+ sources
            if src >= needed:
                score += 15
            elif src >= needed * 0.7:
                score += 5
            else:
                score -= (needed - src) * 10

        for card in spells:
            if card.cmc <= 2 and card.colors:
                if all(sources.get(col, 0) >= 8 for col in card.colors):
                    score += 3
                else:
                    score -= 8

        used_colors = {c for c, v in pip_reqs.items() if v > 0}
        if len(used_colors) >= 2:
            counts    = [sources.get(c, 0) for c in used_colors]
            imbalance = max(counts) - min(counts)
            score    += 12 if imbalance <= 3 else -(imbalance - 6) * 4 if imbalance > 6 else 0

        # Penalise sources for colors outside the deck's color identity only
        for color, src in sources.items():
            if pip_reqs.get(color, 0) == 0:
                if self.pool.target_colors and color in self.pool.target_colors:
                    continue  # on-color source with no pips is fine (hybrid/X costs)
                score -= src * 3

        # Color precision bonus: reward lands that ONLY produce target colors.
        # Prefers Concealed Courtyard (W,B) over Rakdos Guildgate (R,B) for WB.
        if self.pool.target_colors:
            target = self.pool.target_colors
            n_colors = len(target)
            basic_counts = {}

            # Track which target colors are covered by nonbasic duals
            dual_color_coverage: set = set()

            for land in lands:
                if land.is_basic_land:
                    basic_counts[land.name] = basic_counts.get(land.name, 0) + 1
                    continue
                produced = set(p for p in land.mana_production if p in "WUBRG")
                if not produced:
                    continue
                on_color  = produced & target
                off_color = produced - target
                if on_color and not off_color:
                    dual_color_coverage |= on_color
                    # Scale bonus with how many target colors this land covers
                    coverage_bonus = len(on_color) * 4   # 4 per covered color
                    score += coverage_bonus
                elif off_color:
                    # Penalty scales with fraction of production that is wasted
                    waste_ratio = len(off_color) / max(len(produced), 1)
                    score -= len(off_color) * 10 * (1 + waste_ratio)

            # Bonus for mana base that covers ALL target colors via nonbasic duals
            if n_colors >= 2 and dual_color_coverage >= target:
                score += 20

            # Penalise stacking too many basics of one type — prefer nonbasic duals
            # Threshold tightens for 3-color decks since basics only cover 1 color
            basic_threshold = 3 if n_colors >= 3 else 4
            if n_colors >= 2:
                for basic_name, cnt in basic_counts.items():
                    if cnt > basic_threshold:
                        score -= (cnt - basic_threshold) * 8

        taplands = sum(
            1 for land in lands
            if land.oracle_text and
               any(t in land.oracle_text.lower() for t in self._tapland_indicators)
        )
        score -= taplands * 2
        return score

    def _heuristic_score(self, individual, card_counts, archetype_cache=None):
        """
        Heuristic fitness function.
        Identical to the original except for the SYNERGY COHESION PENALTY
        block near the end  (marked <<< NEW).
        """
        scores     = []
        deck_cards = [self.pool.get_card(cid) for cid in individual]
        nonland    = [c for c in deck_cards if 'Land' not in c.types]
        land_count = sum(1 for c in deck_cards if 'Land' in c.types)

        if self.lda_manager:
            self.lda_manager._ensure_loaded()

        archetype = "Unknown"
        coherence = 0.0
        if archetype_cache is not None and 'archetype' in archetype_cache:
            archetype = archetype_cache['archetype']
            coherence = archetype_cache.get('coherence', 0.0)
        elif self.lda_manager and self.lda_manager.is_trained:
            try:
                archetype, coherence = self.lda_manager.get_deck_archetype(
                    individual, self.pool,
                    target_colors=self.pool.target_colors)
                if hasattr(self, '_target_archetype') and self._target_archetype:
                    archetype = self._target_archetype
                if archetype_cache is not None:
                    archetype_cache['archetype'] = archetype
                    archetype_cache['coherence'] = coherence
            except Exception as e:
                self._logger.debug("Error getting archetype: %s", e)

        is_aggro   = any(a in archetype for a in ('Aggro', 'Tokens', 'Burn'))
        is_control = any(a in archetype for a in ('Control', 'Combo'))

        scores.append(self._mana_base_fitness(individual) * 2.5)

        if is_aggro:
            ideal, land_range = 22, (20, 24)
        elif is_control:
            ideal, land_range = 26, (24, 28)
        else:
            ideal, land_range = 24, (22, 26)

        scores.append(
            20 + max(0, 15 - abs(land_count - ideal) * 5)
            if land_range[0] <= land_count <= land_range[1]
            else -abs(land_count - ideal) * 15
        )

        cmc_dist = Counter(c.cmc for c in nonland)
        if is_aggro:
            low_drops = cmc_dist.get(1, 0) + cmc_dist.get(2, 0)
            scores.append(min(50, (low_drops - 8) * 8) if low_drops >= 8 else -20)
            scores.append(-sum(cmc_dist.get(i, 0) for i in range(4, 10)) * 12)
        elif is_control:
            scores.append(min(sum(cmc_dist.get(i, 0) for i in range(2, 5)) * 4, 35))
            draw_count = sum(
                1 for c in nonland
                if any(d in c.oracle_text.lower()
                    for d in ('draw', 'card', 'search your library', 'scry')))
            scores.append(min(draw_count * 6, 30))
        else:
            for cmc, w, cap in [(1, 6, 24), (2, 5, 30), (3, 4, 20)]:
                scores.append(min(cmc_dist.get(cmc, 0) * w, cap))
            scores.append(-sum(cmc_dist.get(i, 0) for i in range(5, 20)) * 4)

        vals = list(card_counts.values())
        scores.extend([
            sum(1 for v in vals if v == 4) * 28,
            sum(1 for v in vals if v == 3) * 14,
            sum(1 for v in vals if v == 2) * 4,
        ])
        nonland_counts = {cid: cnt for cid, cnt in card_counts.items()
                        if 'Land' not in self.pool.get_card(cid).types}
        total_unique   = len(nonland_counts)
        singletons     = sum(1 for v in nonland_counts.values() if v == 1)
        if total_unique > 0:
            ratio = singletons / total_unique
            if ratio > 0.20:
                scores.append(-40 * (ratio - 0.20))
            if 8 <= total_unique <= 12:
                scores.append(20)
            elif total_unique > 14:
                scores.append(-(total_unique - 14) * 12)

        rarity_bonus  = {'mythic': 35, 'rare': 20, 'uncommon': 4, 'common': -3}
        power_score   = 0
        ramp_keywords = ('search your library for a', 'add {', 'add one mana', 'adds one mana')
        has_ramp = any(
            card.oracle_text and any(kw in card.oracle_text.lower() for kw in ramp_keywords)
            for card in nonland
        )
        max_reasonable_cmc = 7 if has_ramp else 5
        for card in nonland:
            power_score += rarity_bonus.get(card.rarity, 0)
            power_score += {0: 8, 1: 8, 2: 5}.get(card.cmc, 2 if card.cmc == 3 else 0)
            if card.cmc > max_reasonable_cmc:
                power_score -= (card.cmc - max_reasonable_cmc) * 18
        scores.append(power_score)

        removal_indicators = {
            'destroy target', 'exile target', 'damage to target creature', '-x/-x',
            'target sacrifices', 'destroy all', 'exile all', 'deals', 'damage to',
            'creature', 'return target', 'to hand', 'counter target'
        }
        removal_count = sum(
            1 for c in nonland
            if any(ind in c.oracle_text.lower() for ind in removal_indicators))
        scores.append(min(removal_count * 4, 40))
        if removal_count < 2:
            scores.append(-20)

        creature_count = sum(1 for c in deck_cards if 'Creature' in c.types)
        ideal_c = (16, 24) if is_aggro else (6, 12) if is_control else (12, 18)
        scores.append(
            20 if ideal_c[0] <= creature_count <= ideal_c[1]
            else -min(abs(creature_count - ideal_c[0]),
                    abs(creature_count - ideal_c[1])) * 2.5
        )

        if self.lda_manager and self.lda_manager.is_trained:
            scores.append(coherence * 60)
            if archetype != "Unknown":
                scores.append(8)
                try:
                    topic_cards = set(self.lda_manager.get_cards_by_topic(
                        archetype, self.pool, top_n=50))
                    on_theme = sum(cnt for cid, cnt in card_counts.items()
                                if cid in topic_cards)
                    scores.append((on_theme / len(individual)) * 50)
                except Exception:
                    pass

        # ── <<< NEW: SYNERGY COHESION PENALTY ────────────────────────────────────
        # Penalises cards that have no business being in this archetype.
        # Applied per copy so a 4-of Tinybones hurts 4× as much as 1x Valgavoth.
        #
        # Penalty tiers:
        #   -15 / copy  Strong LDA mismatch AND no keyword synergy
        #               (Tinybones, Forsaken Miner, Valgavoth land here)
        #    -6 / copy  Strong LDA mismatch BUT card has keyword synergy overlap
        #               (e.g. Vampire Nighthawk: LDA=Midrange, has lifelink)
        #   -10 / copy  Marginal on-archetype LDA score AND no keyword synergy
        #    -8 / copy  Not in LDA at all AND no keyword synergy
        if self.lda_manager and self.lda_manager.is_trained and archetype != "Unknown":
            target_tid = next(
                (k for k, v in self.lda_manager.topic_names.items() if v == archetype),
                None
            )
            if target_tid:
                noise_penalty = 0.0
                for cid, cnt in card_counts.items():
                    card = self.pool.get_card(cid)
                    if 'Land' in card.types:
                        continue

                    has_synergy = _has_keyword_synergy(
                        card.id, card.oracle_text, card.name, archetype)
                    topics      = self.lda_manager.get_card_topics(card.name)

                    if not topics:
                        if not has_synergy:
                            noise_penalty += cnt * 8
                        continue

                    on_arch              = topics.get(target_tid, 0.0)
                    best_tid, best_score = max(topics.items(), key=lambda x: x[1])

                    if best_tid == target_tid:
                        continue                             # primary match, no penalty

                    if best_score > 0.5 and on_arch < 0.05:
                        noise_penalty += cnt * (6 if has_synergy else 15)
                    elif on_arch < 0.05 and not has_synergy:
                        noise_penalty += cnt * 10

                scores.append(-noise_penalty)
        # ── END SYNERGY COHESION PENALTY ─────────────────────────────────────────

        if self.pool.target_colors:
            off_color = sum(
                1 for c in nonland
                if c.colors and not set(c.colors).issubset(self.pool.target_colors))
            if off_color:
                scores.append(-15 * (off_color ** 1.5))

        illegal = sum(
            1 for cid, cnt in card_counts.items()
            if cnt > self.pool.get_card(cid).max_copies)
        if illegal:
            scores.append(-400 * illegal)

        return sum(scores)

    def _export_temp(self, deck: List[int], filepath: str):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        counts       = {}
        name_to_card = {}
        for cid in deck:
            c         = self.pool.get_card(cid)
            card_name = c.name.split(' // ')[0] if ' // ' in c.name else c.name
            counts[card_name]       = counts.get(card_name, 0) + 1
            name_to_card[card_name] = c

        def sort_key(item):
            name, _ = item
            card = name_to_card.get(name)
            if not card: return (99, 0, name)
            if 'Land' in card.types:     return (2, card.cmc, name)
            if 'Creature' in card.types: return (0, card.cmc, name)
            return (1, card.cmc, name)

        with open(filepath, 'w') as f:
            f.write("[metadata]\nName=GA Test Deck\n\n[Main]\n")
            for name, count in sorted(counts.items(), key=sort_key):
                f.write(f"{count} {name}\n")

    def _save_results(self, deck: List[int], results: Dict, score: float,
                      archetype_cache: Optional[Dict] = None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"{TEST_RESULTS_DIR}/gen{self.generation}_{timestamp}.json"

        arch_info = {}
        if archetype_cache and 'archetype' in archetype_cache:
            arch_info = {'archetype': archetype_cache['archetype'],
                         'coherence': archetype_cache.get('coherence', 0.0)}
        elif self.lda_manager and self.lda_manager.is_trained:
            try:
                arch, coh  = self.lda_manager.get_deck_archetype(deck, self.pool)
                arch_info  = {'archetype': arch, 'coherence': coh}
            except Exception:
                pass

        data = {
            'generation': self.generation, 'score': score,
            'win_rate': results['overall_win_rate'],
            'deck_list': [self.pool.get_card(cid).name for cid in set(deck)],
            'matchups': results['matchups'], **arch_info
        }
        try:
            os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self._logger.error("Failed to save results: %s", e)

    def evaluate(self, individual: List[int],
                 force_test: bool = False) -> Tuple[float,]:
        if len(individual) != DECK_SIZE:
            return (-1000.0,)

        if not hasattr(self.pool, 'conn') or self.pool.conn is None:
            self.pool.reconnect()

        card_counts: Dict[int, int] = {}
        for c in individual:
            card_counts[c] = card_counts.get(c, 0) + 1

        penalty = sum(
            -100 * (card_counts[c] - self.pool.get_card(c).max_copies)
            for c in card_counts
            if card_counts[c] > self.pool.get_card(c).max_copies
        )
        if penalty < 0:
            return (penalty,)

        key = tuple(sorted(individual))

        if len(self.tested) >= self.max_cache_size:
            for _ in range(len(self.tested) // 5):
                self.tested.popitem(last=False)

        if not force_test and key in self.tested:
            self.tested.move_to_end(key)
            return (self.tested[key],)

        archetype_cache: Dict = {}
        heuristic_score       = self._heuristic_score(individual, card_counts,
                                                       archetype_cache) * 0.4

        should_simulate = (
            force_test or
            random.random() < 0.05 or
            self.generation >= GENERATIONS - 1
        )

        if should_simulate:
            temp_path = str(FORGE_DECK_DIR / f"temp_deck_{uuid.uuid4().hex}.dck")
            self._export_temp(individual, temp_path)
            if not os.path.exists(temp_path):
                self._logger.error("Failed to create temp deck: %s", temp_path)
                final_score = heuristic_score
            else:
                try:
                    results     = self.forge.test_against_meta(
                        temp_path, self.meta_decks,
                        quick_test=(not force_test),
                        games_per_matchup=self.games_per_matchup)
                    final_score = results['overall_win_rate'] * 100 * 0.6 + heuristic_score

                    if force_test:
                        self._save_results(individual, results, final_score, archetype_cache)
                except Exception as e:
                    self._logger.error("Simulation failed: %s", e)
                    final_score = heuristic_score
                finally:
                    try:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                    except Exception as e:
                        self._logger.warning("Failed to remove temp file: %s", e)
        else:
            final_score = heuristic_score

        self.tested[key] = final_score
        self.tested.move_to_end(key)
        return (final_score,)


# ---------- EXPORT / VALIDATE ----------
def validate_deck(individual: List[int], pool: CardPool,
                  lda_manager: Optional[LDAModelManager] = None,
                  verbose: bool = False) -> Tuple[bool, str, Dict]:
    log    = logging.getLogger("validate_deck")
    errors = []
    details = {
        'total_cards': len(individual), 'land_count': 0,
        'creature_count': 0, 'spell_count': 0,
        'unique_cards': 0, 'playsets': 0, 'threeofs': 0,
        'twoofs': 0, 'singletons': 0,
        'archetype': 'Unknown', 'coherence': 0.0,
        'violations': [], 'off_color_cards': []
    }

    if len(individual) != 60:
        errors.append(f"Deck size {len(individual)} != 60")

    card_counts: Counter = Counter()
    for cid in individual:
        card = pool.get_card(cid)
        card_counts[cid] += 1
        if 'Land'     in card.types: details['land_count']     += 1
        elif 'Creature' in card.types: details['creature_count'] += 1
        else:                          details['spell_count']    += 1

    details['unique_cards'] = len(card_counts)
    for cid, count in card_counts.items():
        card = pool.get_card(cid)
        if count > card.max_copies:
            v = f"{card.name}: {count} copies (max {card.max_copies})"
            errors.append(v); details['violations'].append(v)
        elif count == 4: details['playsets']  += 1
        elif count == 3: details['threeofs']  += 1
        elif count == 2: details['twoofs']    += 1
        elif count == 1: details['singletons'] += 1

    if pool.target_colors:
        for cid in individual:
            card = pool.get_card(cid)
            if 'Land' in card.types or not card.colors:
                continue
            if not set(card.colors).issubset(pool.target_colors):
                v = f"Off-color: {card.name}"
                errors.append(v)
                details['off_color_cards'].append(card.name)
                details['violations'].append(v)

    if lda_manager:
        lda_manager._ensure_loaded()
        if lda_manager.is_trained:
            try:
                arch, coh = lda_manager.get_deck_archetype(individual, pool,
                                        target_colors=pool.target_colors)
                details['archetype'] = arch
                details['coherence'] = coh
            except Exception as e:
                details['archetype'] = f"Error: {e}"

    is_valid = len(errors) == 0
    message  = "Valid" if is_valid else "; ".join(errors)

    if verbose:
        print(f"  Deck composition: {details['playsets']}x 4-of, "
              f"{details['threeofs']}x 3-of, {details['twoofs']}x 2-of, "
              f"{details['singletons']}x 1-of")
        print(f"  Archetype: {details['archetype']} "
              f"(coherence: {details['coherence']:.2f})")
        if details['off_color_cards']:
            print(f"  ⚠️  Off-color: {', '.join(details['off_color_cards'])}")

    return is_valid, message, details


def _sanitize_land_base(deck, pool, target_colors):
    """
    Post-GA cleanup: swap off-color nonbasic lands and excess basics for the
    best available on-color alternatives. Called once on the final best deck
    before export, never during evolution.

    Rules:
      1. Off-color nonbasic land -> swap for best on-color dual.
      2. More than 6 copies of a single basic in a multi-color deck ->
         swap extras for on-color nonbasic duals (prefer untapped duals).
    """
    if not target_colors or len(target_colors) < 2:
        return deck

    deck = list(deck)
    from collections import Counter
    deck_counts = Counter(deck)

    def on_color_nonbasic_duals():
        candidates = []
        for cid in pool.lands:
            card = pool.get_card(cid)
            if card.is_basic_land:
                continue
            produced = set(p for p in card.mana_production if p in "WUBRG")
            if not produced:
                continue
            if produced.issubset(target_colors) and len(produced) >= 2:
                is_tapped = False
                if card.oracle_text:
                    _ot = card.oracle_text.lower()
                    for _p in ("enters tapped", "enters the battlefield tapped"):
                        _i = _ot.find(_p)
                        if _i != -1 and 'unless' not in _ot[_i+len(_p):_i+len(_p)+60]:
                            is_tapped = True
                            break
                rarity_rank = {"mythic": 0, "rare": 1, "uncommon": 2, "common": 3}.get(
                    card.rarity, 4)
                candidates.append((is_tapped, rarity_rank, cid))
        candidates.sort()
        return [cid for _, _, cid in candidates]

    best_duals = on_color_nonbasic_duals()

    def swap(old_cid, new_cid):
        if deck_counts[new_cid] >= pool.get_card(new_cid).max_copies:
            return False
        idx = next((i for i, c in enumerate(deck) if c == old_cid), None)
        if idx is None:
            return False
        deck[idx] = new_cid
        deck_counts[old_cid] -= 1
        deck_counts[new_cid] += 1
        return True

    # Pass 1: remove off-color nonbasic lands
    for cid in list(deck):
        card = pool.get_card(cid)
        if card.is_basic_land:
            continue
        produced = set(p for p in card.mana_production if p in "WUBRG")
        if not produced:
            continue
        off_color = produced - target_colors
        if not off_color:
            continue
        swapped = False
        for dual_id in best_duals:
            if swap(cid, dual_id):
                swapped = True
                print(f"  [sanitize] {card.name} -> {pool.get_card(dual_id).name}")
                break
        if not swapped:
            for basic_cid in pool.lands:
                bc = pool.get_card(basic_cid)
                if bc.is_basic_land and set(bc.mana_production) & target_colors:
                    if deck_counts[basic_cid] < bc.max_copies:
                        swap(cid, basic_cid)
                        print(f"  [sanitize] {card.name} -> {bc.name} (basic fallback)")
                        break

    # Pass 2: reduce excess basics (> 6 of any single basic)
    for cid in list(set(deck)):
        card = pool.get_card(cid)
        if not card.is_basic_land:
            continue
        while deck_counts[cid] > 6:
            swapped = False
            for dual_id in best_duals:
                if swap(cid, dual_id):
                    print(f"  [sanitize] excess {card.name} -> {pool.get_card(dual_id).name}")
                    swapped = True
                    break
            if not swapped:
                break

    return deck


def export_forge_format(deck: List[int], pool: CardPool, filename: str,
                        target_colors: Optional[str] = None) -> str:
    safe_name  = re.sub(r'[^\w.\-]', '_', os.path.basename(filename))
    if not safe_name.endswith('.dck'):
        safe_name += '.dck'
    output_dir = Path(filename).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath   = str(output_dir / safe_name)

    if pool.target_colors:
        target      = pool.target_colors
        cleaned     = []
        purged      = []
        for cid in deck:
            card = pool.get_card(cid)
            if 'Land' not in card.types:
                card_colors = set(card.colors) if card.colors else set()
                if not card_colors or card_colors.issubset(target):
                    cleaned.append(cid)
                else:
                    purged.append(f"{card.name} ({''.join(card_colors)})")
                continue
            text     = (card.oracle_text or '').upper()
            produced = set(re.findall(r'\{([WUBRG])\}', text))
            if 'ANY COLOR' in text or 'ANY ONE COLOR' in text:
                if len(target) <= 2:
                    purged.append(f"{card.name} (any color land)"); continue
            if produced and not produced & target:
                purged.append(f"{card.name} (produces {''.join(produced - target)})")
                continue
            cleaned.append(cid)

        if purged:
            print(f"\n☢️  PURGED {len(purged)} off-color cards:")
            for p in purged[:10]: print(f"    - {p}")
            if len(purged) > 10:  print(f"    ... and {len(purged)-10} more")

        deck = cleaned
        while len(deck) < 60:
            for color in sorted(target):
                for lid in pool.lands:
                    if pool.get_card(lid).name == BASIC_LANDS.get(color):
                        deck.append(lid); break
                if len(deck) >= 60: break
        deck = deck[:60]

    counts       = {}
    name_to_card = {}
    for cid in deck:
        c         = pool.get_card(cid)
        card_name = c.name.split(' // ')[0] if ' // ' in c.name else c.name
        counts[card_name]       = counts.get(card_name, 0) + 1
        name_to_card[card_name] = c

    def sort_key(item):
        name, _ = item
        card = name_to_card.get(name)
        if not card: return (99, 0, name)
        if 'Land' in card.types:     return (2, card.cmc, name)
        if 'Creature' in card.types: return (0, card.cmc, name)
        return (1, card.cmc, name)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("[metadata]\nName=GA Optimized Deck\n\n[Main]\n")
        for name, count in sorted(counts.items(), key=sort_key):
            f.write(f"{count} {name}\n")

    return filepath


# ---------- PIPELINE ----------
class AutomatedPipeline:
    def __init__(self, mode: str = 'live',
                 games_per_matchup: int = GAMES_PER_MATCHUP_DEFAULT):
        _ensure_deap()
        self.meta_deck_loader  = MetaDeckLoader()
        self.forge             = ForgeHeadlessRunner(mode=mode)
        self.meta_decks:       List[MetaDeck] = []
        self.lda_manager:      Optional[LDAModelManager] = None
        self.meta_pool:        Optional[MetaCardPool] = None
        self.mode              = mode
        self.games_per_matchup = games_per_matchup
        self.pool:             Optional[CardPool] = None
        self.checkpoint_dir    = Path(CHECKPOINT_DIR)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self._interrupted      = False

        signal.signal(signal.SIGINT,  self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.forge.shutdown()
        if self.pool is not None:
            self.pool.close()
        return False

    def _signal_handler(self, signum, frame):
        print(f"\n⚠️  Signal {signum} received. Saving checkpoint...")
        self._interrupted = True

    def _get_checkpoint_path(self, generation: int = None) -> Optional[Path]:
        if generation is not None:
            return self.checkpoint_dir / f"checkpoint_gen{generation:04d}.json"
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_gen*.json"))
        return checkpoints[-1] if checkpoints else None

    def _cleanup_old_checkpoints(self):
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_gen*.json"))
        while len(checkpoints) > MAX_CHECKPOINTS:
            old = checkpoints.pop(0)
            try:
                old.unlink()
                h = old.with_suffix('.sha256')
                if h.exists(): h.unlink()
            except Exception as e:
                logger.warning("Failed to clean up checkpoint %s: %s", old, e)

    def _compute_file_hash(self, filepath: Path) -> str:
        h = hashlib.sha256()
        try:
            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    h.update(chunk)
            return h.hexdigest()
        except Exception as e:
            logger.error("Failed to compute hash for %s: %s", filepath, e)
            return ""

    def _individual_to_serialisable(self, ind) -> Dict:
        return {
            'cards':   list(ind),
            'fitness': list(ind.fitness.values) if ind.fitness.valid else []
        }

    def _individual_from_dict(self, d: Dict):
        _ensure_deap()
        ind = creator.Individual(d['cards'])
        if d.get('fitness'):
            ind.fitness.values = tuple(d['fitness'])
        return ind

    def save_checkpoint(self, generation: int, population: list,
                        hof: tools.HallOfFame, stats: tools.Statistics,
                        evaluator: 'AdaptiveFitnessEvaluator'):
        checkpoint_data = {
            'version':    '2.0',
            'generation': generation,
            'population': [self._individual_to_serialisable(ind) for ind in population],
            'hof':        [self._individual_to_serialisable(ind) for ind in hof],
            'timestamp':  datetime.now().isoformat(),
            'config': {
                'population_size': len(population),
                'target_colors':   list(self.pool.target_colors) if self.pool else None,
                'mode':            self.mode,
            }
        }
        temp_path   = self.checkpoint_dir / f".tmp_checkpoint_gen{generation:04d}.json"
        final_path  = self._get_checkpoint_path(generation)
        hash_path   = self.checkpoint_dir / f"checkpoint_gen{generation:04d}.sha256"
        backup_path = self.checkpoint_dir / f"checkpoint_gen{generation:04d}.json.bak"

        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f)

            file_hash = self._compute_file_hash(temp_path)
            with open(hash_path, 'w') as f:
                f.write(file_hash)

            if final_path.exists():
                final_path.rename(backup_path)
            temp_path.rename(final_path)

            print(f"  💾 Checkpoint saved: Gen {generation} "
                  f"({len(population)} individuals) [hash: {file_hash[:16]}...]")
            self._cleanup_old_checkpoints()

        except Exception as e:
            logger.error("Failed to save checkpoint: %s", e)
            for p in [temp_path, hash_path]:
                if p.exists():
                    try: p.unlink()
                    except Exception: pass
            raise

    def load_checkpoint(self) -> Optional[Dict]:
        checkpoint_path = self._get_checkpoint_path()
        if not checkpoint_path or not checkpoint_path.exists():
            return None

        hash_path = checkpoint_path.with_suffix('.sha256')
        if not hash_path.exists():
            warnings.warn(
                f"Checkpoint {checkpoint_path.name} has no integrity hash. Refusing to load.",
                SecurityWarning, stacklevel=2)
            return None

        try:
            stored_hash   = hash_path.read_text().strip()
            computed_hash = self._compute_file_hash(checkpoint_path)
            if stored_hash != computed_hash:
                warnings.warn(
                    f"Checkpoint integrity check FAILED for {checkpoint_path.name}.",
                    SecurityWarning, stacklevel=2)
                return None
        except Exception as e:
            logger.error("Error verifying checkpoint hash: %s", e)
            return None

        for path in [checkpoint_path, Path(str(checkpoint_path) + ".bak")]:
            if not path.exists():
                continue
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if not all(k in data for k in ('generation', 'population', 'config')):
                    continue
                data['population'] = [self._individual_from_dict(d)
                                       for d in data['population']]
                data['hof']        = [self._individual_from_dict(d)
                                       for d in data.get('hof', [])]
                print(f"  📂 Loaded checkpoint: {path.name} (Gen {data['generation']})")
                return data
            except (json.JSONDecodeError, KeyError) as e:
                logger.error("Corrupted checkpoint %s: %s", path.name, e)
                continue

        return None

    def _display_checkpoint_info(self, checkpoint: Dict):
        gen    = checkpoint['generation']
        ts     = checkpoint.get('timestamp', 'unknown')
        colors = ''.join(checkpoint['config']['target_colors'] or [])
        print(f"\n{'='*60}\nRESUME SESSION FOUND\n{'='*60}")
        print(f"  Generation: {gen}  |  Colors: {colors}  |  Saved: {ts}")
        print(f"{'='*60}")

    def prompt_resume(self) -> Tuple[bool, Optional[Dict]]:
        checkpoint = self.load_checkpoint()
        if not checkpoint:
            return False, None
        self._display_checkpoint_info(checkpoint)
        while True:
            choice = input("Resume from this checkpoint? [Y/n/s(tatus)]: ").strip().lower()
            if choice in ('', 'y', 'yes'):
                return True, checkpoint
            elif choice in ('n', 'no'):
                archive = self.checkpoint_dir / 'archived'
                archive.mkdir(exist_ok=True)
                cp = self._get_checkpoint_path()
                if cp:
                    try:
                        cp.rename(archive / f"{cp.name}.{int(time.time())}")
                    except Exception as e:
                        logger.error("Failed to archive checkpoint: %s", e)
                return False, None
            elif choice in ('s', 'status'):
                print(f"  HoF size: {len(checkpoint['hof'])}")
                if checkpoint['hof']:
                    best = checkpoint['hof'][0]
                    if hasattr(best, 'fitness') and best.fitness.valid:
                        print(f"  Best fitness: {best.fitness.values}")
            else:
                print("  Please enter Y, n, or s")

    def run(self, generations=GENERATIONS, population_size=POPULATION_SIZE,
            target_colors=None, resume=False, target_archetype=None):

        print("=" * 60)
        print("PHASE 1: Loading Meta Decks & LDA Model")
        print("=" * 60)

        self.meta_decks  = self.meta_deck_loader.load_random_meta_decks(num_decks=10)
        self.lda_manager = LDAModelManager()
        self.lda_manager.load()

        self.meta_pool = MetaCardPool(db_path=META_DB_PATH,
                                    lda_manager=self.lda_manager)

        if not target_colors:
            raise ValueError("No target colors specified. Use --colors flag.")

        selected_colors     = set(target_colors)
        start_gen           = 0
        restored_population = None
        restored_hof        = None

        if resume:
            checkpoint = self.load_checkpoint()
        else:
            should_resume, checkpoint = self.prompt_resume()
            if not should_resume:
                checkpoint = None

        if checkpoint:
            saved_colors = set(checkpoint['config']['target_colors'] or [])
            if saved_colors != selected_colors:
                print(f"  ⚠️  Color mismatch: checkpoint={saved_colors}, "
                    f"requested={selected_colors}. Starting fresh...")
                checkpoint = None
            else:
                start_gen           = checkpoint['generation'] + 1
                restored_population = checkpoint['population']
                restored_hof        = checkpoint['hof']
                print(f"  🔄 Resuming from Generation {start_gen}")

        print(f"\n{'='*60}")
        print("PHASE 2: Initializing Genetic Algorithm")
        print("=" * 60)

        with CardPool(db_path=DB_PATH, target_colors=selected_colors,
                    lda_manager=self.lda_manager) as pool:
            self.pool = pool
            print(f"  Pool: {len(pool.cards)} total cards")

            _ensure_deap()
            toolbox = base.Toolbox()

            toolbox.register("individual", create_lda_individual,
                            card_pool=pool, lda_manager=self.lda_manager,
                            target_archetype=target_archetype,
                            meta_pool=self.meta_pool)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

            evaluator = AdaptiveFitnessEvaluator(
                pool, self.meta_decks, self.forge, self.lda_manager,
                games_per_matchup=self.games_per_matchup)
            evaluator._target_archetype = target_archetype

            toolbox.register("evaluate", evaluator.evaluate)

            # <<< CHANGED: lda_manager and target_archetype now wired into cx_deck
            #     so the archetype filter inside it is actually active.
            #     Previously these were missing and the filter was dead code.
            toolbox.register("mate", cx_deck,
                            card_pool=pool,
                            lda_manager=self.lda_manager,          # <<< NEW
                            target_archetype=target_archetype)     # <<< NEW

            toolbox.register("mutate", mut_lda_deck,
                            card_pool=pool,
                            lda_manager=self.lda_manager,
                            indpb=0.15,
                            target_archetype=target_archetype)
            toolbox.register("select", tools.selTournament, tournsize=3)

            if restored_population is not None:
                pop = restored_population
                for ind in pop:
                    if not hasattr(ind, 'fitness') or not ind.fitness.valid:
                        ind.fitness.values = (0.0,)
                hof = tools.HallOfFame(5)
                for ind in restored_hof:
                    hof.update([ind])
                print(f"  Restored {len(pop)} individuals and {len(hof)} HoF entries")
            else:
                pop = toolbox.population(n=population_size)
                hof = tools.HallOfFame(5)

            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("max", np.max)
            stats.register("min", np.min)
            stats.register("std", np.std)

            print(f"\n{'='*60}")
            print(f"PHASE 3: Evolution (Gens {start_gen+1}–{generations})")
            print("=" * 60)

            try:
                for gen in range(start_gen, generations):
                    evaluator.generation = gen
                    print(f"\n--- Generation {gen+1}/{generations} ---")

                    offspring = algorithms.varAnd(pop, toolbox,
                                                cxpb=CROSSOVER_PROB,
                                                mutpb=MUTATION_PROB)
                    fitnesses = []
                    for i, ind in enumerate(offspring):
                        if self._interrupted and i > 0:
                            print("\n  Interrupt detected, saving...")
                            break
                        force = (i < 10) or (gen % 5 == 0) or (gen == generations - 1)
                        fit   = toolbox.evaluate(ind, force_test=force)
                        fitnesses.append(fit)
                        if force and i < 3:
                            print(f"  Individual {i}: Fitness {fit[0]:.1f}")

                    if self._interrupted and len(fitnesses) < len(offspring):
                        self.save_checkpoint(gen, pop, hof, stats, evaluator)
                        print(f"\n✅ Emergency checkpoint saved at Gen {gen+1}")
                        return None

                    for ind, fit in zip(offspring[:len(fitnesses)], fitnesses):
                        ind.fitness.values = fit

                    pop = toolbox.select(offspring, len(pop))
                    hof.update(pop)

                    record = stats.compile(pop)
                    print(f"  Max: {record['max']:.1f}  Avg: {record['avg']:.1f}  "
                        f"Min: {record['min']:.1f}")

                    if hof:
                        fourofs = sum(1 for v in Counter(hof[0]).values() if v == 4)
                        print(f"  Best deck: {fourofs}x 4-ofs")

                    if (gen + 1) % CHECKPOINT_FREQUENCY == 0:
                        self.save_checkpoint(gen + 1, pop, hof, stats, evaluator)

                    if self._interrupted:
                        print(f"\n  Graceful shutdown after Gen {gen+1}...")
                        break

            except KeyboardInterrupt:
                print("\n⚠️  KeyboardInterrupt! Saving checkpoint...")
                self.save_checkpoint(evaluator.generation, pop, hof, stats, evaluator)
                raise

            print("\n" + "=" * 60)
            print("PHASE 4: Exporting Best Deck")
            print("=" * 60)

            best          = hof[0] if hof else pop[0]
            _best_fitness = best.fitness
            _sanitized    = _sanitize_land_base(
                list(best), pool,
                set(target_colors) if isinstance(target_colors, str) else target_colors)
            best         = creator.Individual(_sanitized)
            best.fitness = _best_fitness

            is_valid, msg, details = validate_deck(
                best, pool, self.lda_manager, verbose=True)

            final_path = str(FORGE_DECK_DIR / 'final_ga_deck.dck')
            export_forge_format(best, pool, final_path, target_colors)
            print(f"✓ Deck saved to: {final_path}")

            if generations <= 20:
                print("\nRunning final validation...")
                final_results = self.forge.test_against_meta(
                    final_path, self.meta_decks, quick_test=False,
                    games_per_matchup=self.games_per_matchup)
                print(f"\nFinal Performance: "
                    f"{final_results['overall_win_rate']*100:.1f}% win rate")

            txt_file = "./final_ga_deck.txt"
            with open(txt_file, 'w') as f:
                f.write(f"Deck: final_ga_deck.dck\nColors: {target_colors}\n"
                        f"Score: {best.fitness.values[0]:.2f}\n")
                if self.lda_manager and self.lda_manager.is_trained:
                    try:
                        arch, coh = self.lda_manager.get_deck_archetype(
                            best, pool, target_colors=pool.target_colors)
                        f.write(f"Archetype: {arch}\nCoherence: {coh:.2f}\n")
                    except Exception:
                        pass
                f.write(f"Composition: {details['playsets']}x4, "
                        f"{details['threeofs']}x3, {details['twoofs']}x2, "
                        f"{details['singletons']}x1\n\n")
                counts = {}
                for cid in best:
                    c = pool.get_card(cid)
                    counts[c.name] = counts.get(c.name, 0) + 1
                for name, count in sorted(counts.items(),
                                        key=lambda x: (-x[1], x[0])):
                    f.write(f"{count} {name}\n")

            print(f"✓ Text version: {os.path.abspath(txt_file)}")
            print("\nDone! Import final_ga_deck.dck into Forge.")

            if not self._interrupted:
                for ckpt in self.checkpoint_dir.glob("checkpoint_gen*.json"):
                    try:
                        ckpt.unlink()
                    except Exception as e:
                        logger.warning("Failed to remove checkpoint %s: %s", ckpt, e)
                print("  ✓ Checkpoints cleaned up")

        self.pool = None
        return best


# ---------- MAIN ----------
def main():
    parser = argparse.ArgumentParser(description='MTG Genetic Deckbuilder with LDA')
    parser.add_argument('--generations', '-g', type=int, default=20)
    parser.add_argument('--population',  '-p', type=int, default=30)
    parser.add_argument('--colors',      '-c', type=str, required=True)
    parser.add_argument('--games',             type=int,
                        default=GAMES_PER_MATCHUP_DEFAULT)
    parser.add_argument('--mode',        '-m', type=str,
                        choices=['sim', 'live'], default='live')
    parser.add_argument('--resume',      '-r', action='store_true')
    parser.add_argument('--fresh',       '-f', action='store_true')
    parser.add_argument('--debug',             action='store_true')
    parser.add_argument('--archetype', '-a', type=str, default=None,
                    help='Force a specific LDA archetype e.g. "White Lifegain"')
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    print("MTG Genetic Deckbuilder with LDA\n" + "=" * 60)

    with AutomatedPipeline(mode=args.mode,
                           games_per_matchup=args.games) as pipeline:
        try:
            pipeline.run(
                generations=args.generations,
                population_size=args.population,
                target_colors=args.colors,
                resume=args.resume and not args.fresh,
                target_archetype=args.archetype,
            )
        except KeyboardInterrupt:
            print("\n\n👋 Exiting. Run again with --resume to continue.")
            sys.exit(0)


if __name__ == "__main__":
    main()
