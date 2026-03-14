#!/usr/bin/env python3
"""
MTG LDA Model Trainer
Trains a topic model on decks from decks_cache.db and saves lda_model.json
for use by the genetic deckbuilder.

Usage:
    python3 train_lda.py --all-decks
    python3 train_lda.py --all-decks --topics 7
    python3 train_lda.py --all-decks --topics auto

FIXES APPLIED:
  1. _normalize_name — colon stripping added to match genetic_deckbuilder.py
  3. _detect_strategy — sample widened from top-10 to top-30; weighted-mean
     stats replace the unweighted sum so high-loading decks count more
  4. Duplicate topic merging — duplicate pairs are now collapsed before saving;
     card_topics remapped to the surviving (higher-variance) topic
  5. card_topics weight correction — raw model.components_ weights are divided
     by each feature's document frequency before normalising, compensating for
     the MAX_CARD_TOKEN_COUNT cap that suppresses staple-4-of dominance during
     training but otherwise leaves their absolute column sums inflated
"""

import json
import math
import re
import sqlite3
import argparse
import numpy as np
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# ---------- CONFIG ----------
DB_PATH        = Path('./decks_cache.db')
LDA_MODEL_PATH = './lda_model.json'
FORGE_DECK_DIR = Path('/home/toaster/.forge/decks/constructed')
LDA_TOPICS     = 'auto'

JSD_DUPLICATE_THRESHOLD = 0.05

# Basic lands should never influence topic content
BASIC_LANDS = {'Plains', 'Island', 'Swamp', 'Mountain', 'Forest', 'Wastes'}
BASIC_LANDS_NORM = {n.lower() for n in BASIC_LANDS}

# Cap card name token repetition at this value so staple 4-ofs don't dominate
MAX_CARD_TOKEN_COUNT = 4

# FIX #3: widen the per-topic deck sample used for strategy detection
STRATEGY_SAMPLE_SIZE = 30


# ---------- DATA CLASSES ----------
class MetaDeck:
    __slots__ = ('name', 'colors', 'card_names', 'is_meta')

    def __init__(self, name: str, colors: str, card_names: List[str], is_meta: bool):
        self.name       = name
        self.colors     = colors
        self.card_names = card_names
        self.is_meta    = is_meta


class CardData:
    __slots__ = ('name', 'mana_cost', 'cmc', 'colors', 'type_line',
                 'oracle_text', 'keywords', 'rarity', 'is_basic_land')

    def __init__(self, name, mana_cost, cmc, colors, type_line,
                 oracle_text, keywords, rarity, is_basic_land):
        self.name          = name
        self.mana_cost     = mana_cost or ''
        self.cmc           = cmc or 0
        self.colors        = colors or []
        self.type_line     = type_line or ''
        self.oracle_text   = oracle_text or ''
        self.keywords      = keywords or []
        self.rarity        = rarity or ''
        self.is_basic_land = bool(is_basic_land)


# ---------- DATABASE LOADERS ----------
class CardPool:
    """Loads card data from decks_cache.db."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path       = db_path
        self.cards:         Dict[str, CardData] = {}
        self.cards_by_name: Dict[str, CardData] = {}
        self._load_cards()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self.cards.clear()
        self.cards_by_name.clear()

    @staticmethod
    def _normalize_name(name: str) -> str:
        # FIX #1: added ':' stripping so this matches genetic_deckbuilder.py exactly.
        # Previously train_lda.py stripped ':' but deckbuilder did not, causing
        # lookup misses for cards like "Nissa, Resurgent Animist".
        return (name.lower().strip()
                .replace(' ', '_')
                .replace('-', '_')
                .replace("'", '')
                .replace(':', '')
                .replace(',', ''))

    def _load_cards(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name, mana_cost, cmc, colors, type_line,
                       oracle_text, keywords, rarity, is_basic_land
                FROM cards
            """)
            for row in cursor.fetchall():
                name, mana_cost, cmc, colors_json, type_line, \
                    oracle_text, keywords_json, rarity, is_basic = row
                try:
                    colors   = json.loads(colors_json)   if colors_json   else []
                    keywords = json.loads(keywords_json) if keywords_json else []
                except (json.JSONDecodeError, TypeError):
                    colors, keywords = [], []

                card = CardData(name, mana_cost, cmc, colors, type_line,
                                oracle_text, keywords, rarity, is_basic)
                norm = self._normalize_name(name)
                self.cards[norm]         = card
                self.cards_by_name[norm] = card

    def derive_deck_colors(self, card_names: List[str]) -> str:
        """
        Derive actual color identity from card data rather than trusting
        the deck name heuristic in the scraper.
        """
        color_set: Set[str] = set()
        for raw_name in card_names:
            if raw_name in BASIC_LANDS:
                continue
            norm = self._normalize_name(raw_name)
            card = self.cards_by_name.get(norm)
            if card and card.colors:
                color_set.update(card.colors)
        return ''.join(c for c in 'WUBRG' if c in color_set)


class DeckLoader:
    """Loads decks from decks_cache.db."""

    def __init__(self, db_path: Path = DB_PATH, deck_dir: Path = FORGE_DECK_DIR):
        self.db_path  = db_path
        self.deck_dir = deck_dir

    def load_decks(self, max_decks: Optional[int] = None,
                   meta_only: bool = False) -> List[MetaDeck]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if meta_only:
                cursor.execute("SELECT name, colors, card_names, is_meta "
                               "FROM decks WHERE is_meta = 1")
            else:
                cursor.execute("SELECT name, colors, card_names, is_meta FROM decks")
            rows = cursor.fetchall()

        print(f"Found {len(rows)} decks in DB, "
              f"using {'all' if not meta_only else 'meta only'}")

        decks = []
        for name, colors, card_names_json, is_meta in rows:
            try:
                card_names = json.loads(card_names_json) if card_names_json else []
            except (json.JSONDecodeError, TypeError):
                continue
            if len(card_names) >= 20:
                decks.append(MetaDeck(name, colors or '', card_names, bool(is_meta)))

        if max_decks:
            decks = decks[:max_decks]

        print(f"Loaded {len(decks)} valid decks (min 20 cards)")
        return decks


# ---------- LDA TRAINER ----------
class LDATrainer:

    def __init__(self, n_topics=7):
        self.n_topics     = n_topics  # may be 'auto'; resolved in train()
        self.model        = None
        self.vectorizer   = None
        self.feature_names:   List[str]                    = []
        self.card_topics:     Dict[str, Dict[str, float]]  = {}
        self.topic_names:     Dict[str, str]               = {}
        self.duplicate_pairs: List[Tuple[int, int, float]] = []
        # FIX #4: track topic merges so card_topics can be remapped
        self._merged_topics:  Dict[int, int]               = {}   # loser -> survivor
        self.is_trained   = False

    @staticmethod
    def _normalize_name(name: str) -> str:
        # FIX #1: must exactly match CardPool._normalize_name and
        # genetic_deckbuilder.py's LDAModelManager._normalize_name.
        return (name.lower().strip()
                .replace(' ', '_')
                .replace('-', '_')
                .replace("'", '')
                .replace(':', '')
                .replace(',', ''))

    @staticmethod
    def _auto_topic_count(n_decks: int) -> int:
        return max(3, min(12, math.floor(math.sqrt(n_decks / 3))))

    # ------------------------------------------------------------------ #
    #  Feature extraction                                                  #
    # ------------------------------------------------------------------ #
    def _extract_features(self, deck: MetaDeck, card_pool: CardPool) -> str:
        """
        Convert a deck into a bag-of-words token string for CountVectorizer.

        Token prefixes:
          COLOR_X  — card colour identity
          TYPE_X   — card type (Creature, Instant, …)
          MECH_X   — gameplay mechanic
          CMC_N    — mana cost bucket
          <name>   — normalised card name (basic lands excluded)

        Card name tokens are capped at MAX_CARD_TOKEN_COUNT so that
        universal 4-ofs (dual lands, staple spells) don't dominate topics.
        Basic lands are excluded entirely from all token types.
        """
        card_counts: Counter = Counter(deck.card_names)
        tokens: List[str] = []

        for raw_name, count in card_counts.items():
            # Skip basic lands completely — they carry no archetype signal
            if raw_name in BASIC_LANDS:
                continue

            norm = self._normalize_name(raw_name)
            card = card_pool.cards_by_name.get(norm)
            if not card:
                card = card_pool.cards_by_name.get(
                    self._normalize_name(raw_name.replace('_', ' ')))

            # Cap name tokens to reduce staple-4-of dominance
            capped_count = min(count, MAX_CARD_TOKEN_COUNT)
            tokens += [norm] * capped_count

            if not card:
                continue

            oracle = card.oracle_text.lower()
            colors = card.colors or []
            types  = card.type_line or ''
            cmc    = int(card.cmc or 0)

            for c in colors:
                tokens += [f"COLOR_{c.upper()}"] * capped_count

            for t in ['Creature', 'Instant', 'Sorcery', 'Enchantment',
                      'Artifact', 'Land', 'Planeswalker']:
                if t in types:
                    tokens += [f"TYPE_{t}"] * capped_count

            for mech in self._extract_mechanics(oracle):
                tokens += [f"MECH_{mech}"] * capped_count

            tokens += [f"CMC_{cmc}"] * capped_count

        return ' '.join(tokens)

    @staticmethod
    def _extract_mechanics(text: str) -> Set[str]:
        """
        Maps oracle text substrings to mechanic tokens emitted into the
        bag-of-words feature space.  Each token becomes a MECH_X feature
        that LDA can use to separate archetypes.

        Organised by archetype so gaps are easy to spot.  Longer/more
        specific phrases must appear BEFORE shorter ones that would
        subsume them (e.g. 'whenever you gain life' before 'gain life')
        — though since we use a set this only matters for readability.
        """
        if not text:
            return set()
        mechs = set()

        # ── LIFEGAIN ─────────────────────────────────────────────────────
        # Only emit 'lifegain' when the card's PRIMARY function involves
        # gaining life.  Phrases like "you may pay 2 life" (shock lands),
        # "deals damage equal to your life total" (burn), or reminder text
        # containing "life" must not trigger this.
        #
        # Strategy: strip reminder text in parentheses first, then require
        # at least one of the high-confidence phrases.  'you gain' is kept
        # but gated so it must be followed by a life-related word within
        # the same clause — the regex handles this.
        _clean = re.sub(r'\([^)]*\)', '', text)   # strip reminder text

        _LIFEGAIN_STRONG = (
            'whenever you gain life',           # payoff trigger (Pridemate, Vito)
            'the next time you would gain life', # replacement effect
            'life you gain',                    # "equal to the life you gain"
            'gains lifelink',                   # aura/pump granting lifelink
            'lifelink',                         # keyword on card itself
            'each opponent loses',              # drain mirror (Vito, Sanguine Bond)
            'extort',                           # per-spell drain keyword
        )
        _lifegain_found = any(phrase in _clean for phrase in _LIFEGAIN_STRONG)

        # 'you gain' and 'gain life' need context — must not be inside a
        # cost clause ("pay life", "lose life as a cost") and must refer
        # to gaining, not paying.  Require the word 'life' to follow
        # within 6 words and not be preceded by 'pay' or 'lose'.
        if not _lifegain_found:
            if re.search(r'you gain \d+ life', _clean):
                _lifegain_found = True
            elif 'gain life' in _clean and 'pay' not in _clean[:_clean.find('gain life')].split()[-3:]:
                _lifegain_found = True

        if _lifegain_found:
            mechs.add('lifegain')

        # ── SACRIFICE / ARISTOCRATS ───────────────────────────────────────
        SACRIFICE = (
            'whenever a creature you control dies',
            'whenever another creature dies',
            'when this creature dies',
            'when ~ dies',                  # templated shorthand
            'whenever you sacrifice',
            'sacrifice a creature',
            'sacrifice another',
        )
        for phrase in SACRIFICE:
            if phrase in text:
                mechs.add('death_trigger')
                break

        if 'sacrifice' in text:
            mechs.add('sacrifice')

        # ── TOKENS ───────────────────────────────────────────────────────
        TOKEN = (
            'create',           # "create a 1/1 white token"
            'amass',            # Zombie army tokens
            'fabricate',        # artifact token or +1/+1
            'afterlife',        # Spirit tokens on death
            'embalm',           # token from graveyard
            'eternalize',       # 4/4 token from graveyard
            'populate',         # copy a token
            'convoke',          # tap creatures to cast — token sink
        )
        for phrase in TOKEN:
            if phrase in text:
                mechs.add('token_generation')
                break

        # anthem effect — "creatures you control get +1" — token payoff
        if 'creatures you control get +' in text:
            mechs.add('anthem')

        # ── AGGRO ─────────────────────────────────────────────────────────
        AGGRO_KW = (
            'haste', 'first strike', 'double strike', 'menace', 'trample',
            'prowess',          # spell-triggered pump
            'exert',            # tap-based aggro keyword
            'dash',             # red aggro evasion
            'battalion',        # Boros attack-matters
            'raid',             # attack-based payoff
            'spectacle',        # damage-gated cost reduction
        )
        for phrase in AGGRO_KW:
            if phrase in text:
                mechs.add('aggro_keyword')
                break

        AGGRO_TRIGGERS = (
            'whenever this creature attacks',
            'one or more creatures attack',
            'whenever ~ attacks',
            'combat damage to a player',
        )
        for phrase in AGGRO_TRIGGERS:
            if phrase in text:
                mechs.add('combat')
                break

        if 'attack' in text:
            mechs.add('combat')

        # ── CONTROL ──────────────────────────────────────────────────────
        if 'counter target' in text:
            mechs.add('counterspell')

        REMOVAL = (
            'destroy target',
            'exile target',
            "can't attack",
            "can't block",
            'return target',        # bounce
            'tap target',           # soft control
        )
        for phrase in REMOVAL:
            if phrase in text:
                mechs.add('removal')
                break

        CONTROL_MISC = (
            'scry',
            'surveil',
            'foretell',
            'cycling',
            'end of your turn',
            'at the beginning of your upkeep',
            'until end of turn',
        )
        for phrase in CONTROL_MISC:
            if phrase in text:
                mechs.add('control_misc')
                break

        # ── CARD DRAW ─────────────────────────────────────────────────────
        DRAW = (
            'draw three cards',
            'draw two cards',
            'draw a card',
            'draw cards',
        )
        for phrase in DRAW:
            if phrase in text:
                mechs.add('card_draw')
                break

        # ── RAMP ──────────────────────────────────────────────────────────
        RAMP = (
            'search your library for a land',
            'put a land',
            'additional land',      # "you may play an additional land"
            'put onto the battlefield',  # Cultivate / Kodama's Reach
            'basic land card',
            'untap target land',    # land-untap ramp (Selvala variants)
        )
        for phrase in RAMP:
            if phrase in text:
                mechs.add('ramp')
                break

        if 'add {' in text:
            mechs.add('mana_generation')

        RAMP_KW = ('improvise', 'convoke')
        for kw in RAMP_KW:
            if kw in text:
                mechs.add('alternate_cost_ramp')
                break

        # ── GRAVEYARD / RECURSION ─────────────────────────────────────────
        GRAVE_RECURSION = (
            'from your graveyard',
            'return target creature card',
            'return target card',
        )
        for phrase in GRAVE_RECURSION:
            if phrase in text:
                mechs.add('recursion')
                break

        GRAVE_KW = (
            'flashback',
            'unearth',
            'escape',           # Theros graveyard keyword
            'aftermath',        # split graveyard spells
            'jump-start',       # Izzet flashback variant
            'disturb',          # white/blue graveyard transform
            'delve',            # exile-from-graveyard cost
            'threshold',        # 7-card graveyard payoff
            'delirium',         # 4 card-type payoff
            'undergrowth',      # creature-count-in-graveyard payoff
        )
        for kw in GRAVE_KW:
            if kw in text:
                mechs.add('graveyard_keyword')
                break

        if 'graveyard' in text:
            mechs.add('graveyard')

        if 'mill' in text:
            mechs.add('mill')

        # ── +1/+1 COUNTERS ────────────────────────────────────────────────
        COUNTER_KW = (
            '+1/+1 counter',
            'proliferate',
            'bolster',
            'adapt',
            'evolve',
            'modular',          # artifact counter transfer
            'reinforce',        # put counters from hand
            'graft',            # counter transfer
            'outlast',          # tap to add counter
            'mentor',           # attack-based counter
            'training',         # attack-with-bigger-creature counter
            'support',          # put counters on others
            'sunburst',         # multicolor counter accumulation
            'charge counter',   # artifact counter subtype
        )
        for kw in COUNTER_KW:
            if kw in text:
                mechs.add('counters')
                break

        # ── LANDFALL ──────────────────────────────────────────────────────
        LANDFALL = (
            'landfall',
            'whenever a land enters',
            'whenever you play a land',
        )
        for phrase in LANDFALL:
            if phrase in text:
                mechs.add('landfall')
                break

        # ── MIDRANGE / VALUE ──────────────────────────────────────────────
        MIDRANGE = (
            'enters the battlefield',
            'when ~ enters',
            'cascade',
            'evoke',
            'champion',
        )
        for phrase in MIDRANGE:
            if phrase in text:
                mechs.add('etb')
                break

        # ── EVASION / COMBAT KEYWORDS (shared) ───────────────────────────
        for kw in ('flying', 'deathtouch', 'hexproof', 'indestructible',
                   'vigilance', 'reach', 'flash', 'ward',
                   'kicker', 'discard'):
            if kw in text:
                mechs.add(kw)

        # ── MISC ──────────────────────────────────────────────────────────
        if 'deals damage' in text:
            mechs.add('damage')

        # Regex catch for "gain N life" (Soul Warden, etc.)
        if re.search(r'gain \d+ life', text):
            mechs.add('lifegain')

        return mechs

    # ------------------------------------------------------------------ #
    #  Model fitting                                                       #
    # ------------------------------------------------------------------ #
    def _fit_model(self, doc_term_matrix,
                   n_topics: int) -> LatentDirichletAllocation:
        model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=500,
            learning_method='batch',
            doc_topic_prior=0.5,
            topic_word_prior=0.01,
            verbose=0,
        )
        model.fit(doc_term_matrix)
        return model

    @staticmethod
    def _jsd(p: np.ndarray, q: np.ndarray) -> float:
        p = p / (p.sum() + 1e-10)
        q = q / (q.sum() + 1e-10)
        m = 0.5 * (p + q)

        def kl(a, b):
            mask = a > 0
            return float(np.sum(a[mask] * np.log(a[mask] / (b[mask] + 1e-10))))

        return float(np.sqrt(0.5 * kl(p, m) + 0.5 * kl(q, m)))

    def _find_duplicate_topics(self) -> List[Tuple[int, int, float]]:
        dupes = []
        n = self.model.n_components
        for i in range(n):
            for j in range(i + 1, n):
                jsd = self._jsd(self.model.components_[i],
                                self.model.components_[j])
                if jsd < JSD_DUPLICATE_THRESHOLD:
                    dupes.append((i, j, jsd))
        return dupes

    # ------------------------------------------------------------------ #
    #  FIX #4: duplicate topic merging                                    #
    # ------------------------------------------------------------------ #
    def _merge_duplicate_topics(self, doc_term_matrix) -> Dict[int, int]:
        """
        For each duplicate pair (i, j, jsd), keep the topic with higher
        variance across documents (i.e. the more discriminative one) and
        remap the loser to the survivor.

        Returns a dict {loser_idx: survivor_idx} for use when building
        card_topics.  The surviving topic's word distribution is updated
        to the average of the two so no signal is lost.

        Only the *transitive closure* of duplicates is merged — if topics
        A≈B and B≈C we collapse all three to the single best survivor.
        """
        if not self.duplicate_pairs:
            return {}

        doc_topic_matrix = self.model.transform(doc_term_matrix)

        # Build connected components of duplicate pairs (union-find)
        parent: Dict[int, int] = {i: i for i in range(self.model.n_components)}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i, j, _ in self.duplicate_pairs:
            union(i, j)

        # Group topics by component
        from collections import defaultdict
        components: Dict[int, List[int]] = defaultdict(list)
        for t in range(self.model.n_components):
            components[find(t)].append(t)

        merged: Dict[int, int] = {}   # loser -> survivor

        for root, members in components.items():
            if len(members) < 2:
                continue

            # Survivor = topic with highest variance in doc_topic_matrix
            # (most discriminative — best at separating decks from each other)
            variances = {t: float(np.var(doc_topic_matrix[:, t])) for t in members}
            survivor  = max(variances, key=variances.__getitem__)

            losers = [t for t in members if t != survivor]
            for loser in losers:
                merged[loser] = survivor
                # Average the word distributions so the survivor inherits
                # any unique signal the loser carried
                self.model.components_[survivor] = (
                    self.model.components_[survivor] +
                    self.model.components_[loser]
                ) / 2.0
                print(f"  [merge] topic_{loser} → topic_{survivor} "
                      f"(variance {variances[loser]:.4f} → {variances[survivor]:.4f})")

        return merged

    # ------------------------------------------------------------------ #
    #  Strategy detection from deck composition stats                     #
    # ------------------------------------------------------------------ #
    def _detect_strategy(self, top_decks: List[MetaDeck],
                         card_pool: CardPool,
                         topic_idx: Optional[int] = None,
                         # FIX #3: accept per-deck loading weights
                         deck_weights: Optional[List[float]] = None) -> str:
        """
        Classify a group of decks into a strategy archetype.

        FIX #3 changes:
          - Sample widened from 10 to STRATEGY_SAMPLE_SIZE (30) decks
          - All composition counters are now *weighted* by deck_weights
            (the topic loading for each deck), so decks that strongly
            represent the topic contribute proportionally more signal
          - This prevents a single off-theme deck in the top-10 from
            flipping the strategy label

        Combines two signals:
          1. Weighted deck composition stats
          2. LDA topic word distribution mechanic token votes
        """
        # --- Signal 2: LDA topic word votes ---
        # Use actual component weights (not occurrence counts) so that
        # most_common() reflects the true strength of each mechanic signal.
        lda_mech_votes: Counter = Counter()
        if topic_idx is not None and self.model is not None:
            top_feat_indices = self.model.components_[topic_idx].argsort()[-30:][::-1]
            for fi in top_feat_indices:
                feat = self.feature_names[fi]
                if feat.startswith('MECH_'):
                    mech = feat[5:]
                    lda_mech_votes[mech] += self.model.components_[topic_idx][fi]

        if not top_decks:
            return self._strategy_from_lda_votes(lda_mech_votes)

        # Normalise weights (default: equal weight)
        if deck_weights is None or len(deck_weights) != len(top_decks):
            deck_weights = [1.0] * len(top_decks)
        total_weight = sum(deck_weights) or 1.0

        # --- Signal 1: Weighted deck composition stats ---

        total_cards       = 0.0
        total_cmc         = 0.0
        creature_count    = 0.0
        spell_count       = 0.0
        landfall_count    = 0.0
        token_count       = 0.0
        anthem_count      = 0.0
        graveyard_count   = 0.0
        ramp_count        = 0.0
        lifegain_count    = 0.0
        sacrifice_count   = 0.0
        death_trig_count  = 0.0
        counter_count     = 0.0
        aggro_count       = 0.0
        control_count     = 0.0
        mill_count        = 0.0

        # Pre-compile patterns used in the inner loop
        _GRAVE_KW    = ('flashback', 'unearth', 'escape', 'aftermath',
                        'jump-start', 'disturb', 'delve', 'threshold',
                        'delirium', 'undergrowth')
        _COUNTER_KW  = ('+1/+1 counter', 'proliferate', 'bolster', 'adapt',
                        'evolve', 'modular', 'reinforce', 'graft', 'outlast',
                        'mentor', 'training', 'support', 'sunburst',
                        'charge counter')
        _AGGRO_KW    = ('prowess', 'exert', 'dash', 'battalion', 'raid',
                        'spectacle', 'haste', 'first strike', 'double strike',
                        'menace')
        _TOKEN_KW    = ('amass', 'fabricate', 'afterlife', 'embalm',
                        'eternalize', 'populate')
        _DEATH_TRIG  = ('whenever a creature you control dies',
                        'whenever another creature dies',
                        'when this creature dies', 'whenever you sacrifice')
        _RAMP_KW     = ('search your library for a land', 'put a land',
                        'additional land', 'basic land card',
                        'put onto the battlefield')
        _CONTROL_KW  = ('counter target', 'destroy target', 'exile target',
                        "can't attack", "can't block", 'scry', 'surveil',
                        'foretell', 'cycling')

        for deck, w in zip(top_decks, deck_weights):
            card_freq: Counter = Counter(deck.card_names)
            for raw_name, qty in card_freq.items():
                if raw_name in BASIC_LANDS:
                    continue
                norm = self._normalize_name(raw_name)
                card = card_pool.cards_by_name.get(norm)
                if not card:
                    card = card_pool.cards_by_name.get(
                        self._normalize_name(raw_name.replace('_', ' ')))
                if not card or card.is_basic_land:
                    continue

                types  = card.type_line or ''
                oracle = card.oracle_text.lower()
                cmc    = float(card.cmc or 0)
                wqty   = w * qty   # FIX #3: weight by topic loading

                total_cards += wqty
                total_cmc   += cmc * wqty

                if 'Creature' in types:
                    creature_count += wqty
                if 'Instant' in types or 'Sorcery' in types:
                    spell_count += wqty

                # Landfall
                if 'landfall' in oracle or 'whenever a land enters' in oracle:
                    landfall_count += wqty

                # Tokens
                if (oracle.lstrip().startswith('create') and 'token' in oracle):
                    token_count += wqty
                elif any(kw in oracle for kw in _TOKEN_KW):
                    token_count += wqty
                if 'creatures you control get +' in oracle:
                    anthem_count += wqty

                # Graveyard
                if ('from your graveyard' in oracle or
                        any(kw in oracle for kw in _GRAVE_KW)):
                    graveyard_count += wqty

                # Mill
                if 'mill' in oracle or 'put the top' in oracle:
                    mill_count += wqty

                # Ramp — non-land cards only; dual lands produce mana but
                # are mana-fixing, not ramp.  Counting 'add {' on lands
                # inflates ramp_pct for every dual-land-heavy control deck.
                if 'Land' not in types:
                    if ('add {' in oracle or
                            any(kw in oracle for kw in _RAMP_KW)):
                        ramp_count += wqty

                # Lifegain — require strong signal, same logic as _extract_mechanics.
                # Strip reminder text first to avoid shock-land false positives.
                _oracle_clean = re.sub(r'\([^)]*\)', '', oracle)
                _LIFEGAIN_STRONG = (
                    'whenever you gain life', 'the next time you would gain life',
                    'life you gain', 'gains lifelink', 'lifelink',
                    'each opponent loses', 'extort',
                )
                _lg = (any(p in _oracle_clean for p in _LIFEGAIN_STRONG) or
                       bool(re.search(r'you gain \d+ life', _oracle_clean)) or
                       ('gain life' in _oracle_clean and
                        'pay' not in _oracle_clean[:_oracle_clean.find('gain life')].split()[-3:]))
                if _lg:
                    lifegain_count += wqty

                # Sacrifice / Aristocrats
                if 'sacrifice' in oracle:
                    sacrifice_count += wqty
                if any(phrase in oracle for phrase in _DEATH_TRIG):
                    death_trig_count += wqty

                # +1/+1 counters
                if any(kw in oracle for kw in _COUNTER_KW):
                    counter_count += wqty

                # Aggro keywords
                if any(kw in oracle for kw in _AGGRO_KW):
                    aggro_count += wqty

                # Control
                if any(kw in oracle for kw in _CONTROL_KW):
                    control_count += wqty

        if total_cards == 0:
            return self._strategy_from_lda_votes(lda_mech_votes)

        avg_cmc          = total_cmc         / total_cards
        creature_pct     = creature_count    / total_cards
        spell_pct        = spell_count       / total_cards
        landfall_pct     = landfall_count    / total_cards
        token_pct        = (token_count + anthem_count * 0.5) / total_cards
        grave_pct        = graveyard_count   / total_cards
        ramp_pct         = ramp_count        / total_cards
        lifegain_pct     = lifegain_count    / total_cards
        sacrifice_pct    = sacrifice_count   / total_cards
        death_trig_pct   = death_trig_count  / total_cards
        counter_pct      = counter_count     / total_cards
        aggro_kw_pct     = aggro_count       / total_cards
        control_pct      = control_count     / total_cards
        mill_pct         = mill_count        / total_cards

        # ── Composition-based rules, ordered by specificity ──────────────
        # Mill — very distinctive, check first
        if mill_pct > 0.08:
            return 'Mill'
        # Landfall
        if landfall_pct > 0.06:
            return 'Landfall'
        # Aristocrats — death triggers are more specific than bare sacrifice
        if death_trig_pct > 0.08:
            return 'Aristocrats'
        # Tokens
        if token_pct > 0.15:
            return 'Tokens'
        # Graveyard
        if grave_pct > 0.10:
            return 'Graveyard'
        # Sacrifice
        if sacrifice_pct > 0.18:
            return 'Sacrifice'
        # Lifegain — checked BEFORE counters because lifegain creatures
        # (Pridemate, Essence Channeler) accumulate +1/+1 counters as a
        # side-effect, causing counter_pct to fire first and mislabel the topic.
        #
        # LDA VETO: if the LDA word distribution's top signal is a strong
        # non-lifegain mechanic (counterspell, graveyard, aggro_keyword,
        # card_draw), that overrides the composition stat.  This handles
        # topics like Izzet Control where off-theme lifegain creatures in the
        # sampled decks inflate lifegain_pct even after the loading threshold.
        # LDA VETO: if the LDA word distribution contains strong signals for a
        # non-lifegain archetype in its top features, those override the
        # composition stat.  Checks top-3 mechs (not just #1) because
        # mana_generation often wins the frequency race on dual-land-heavy
        # topics even though card_draw / counterspell are the real signal.
        _LIFEGAIN_VETO = {'counterspell', 'card_draw', 'removal',
                          'graveyard', 'graveyard_keyword', 'recursion',
                          'aggro_keyword', 'death_trigger',
                          'landfall', 'mill', 'token_generation'}
        _top3_mechs = {m for m, _ in lda_mech_votes.most_common(3)}
        _lifegain_vetoed = bool(_top3_mechs & _LIFEGAIN_VETO)
        if lifegain_pct > 0.12 and creature_pct > 0.30 and not _lifegain_vetoed:
            return 'Lifegain'
        # Counters — only reached if lifegain didn't win
        if counter_pct > 0.12:
            return 'Counters'
        # Ramp
        if ramp_pct > 0.10 and avg_cmc > 2.8:
            return 'Ramp'
        # Control — high spell % or high control-keyword density
        if (spell_pct > 0.35 and avg_cmc > 2.6) or control_pct > 0.20:
            return 'Control'
        # Aggro — low curve, high creature %, or dense aggro keywords
        if (avg_cmc < 2.6 and creature_pct > 0.38) or aggro_kw_pct > 0.15:
            return 'Aggro'

        # Ambiguous — break tie with LDA mechanic votes
        if lda_mech_votes:
            return self._strategy_from_lda_votes(lda_mech_votes)

        return 'Midrange'

    @staticmethod
    def _strategy_from_lda_votes(votes: Counter) -> str:
        """
        Map the most prominent LDA MECH_ token to a strategy label.
        Covers all mechanic tokens emitted by _extract_mechanics.

        Ordering is critical — more archetype-specific tokens must appear
        before broader ones.  In particular:
          - mana_generation fires on every dual land so must be last resort
          - card_draw / counterspell are genuine control signals and win
            over mana_generation when both appear in an Izzet/Dimir topic
        """
        # Process votes in specificity order rather than frequency order:
        # iterate through the priority list and return the first strategy
        # whose mechanic appears in the votes at all (count > 0).
        PRIORITY = [
            # ── highest specificity — unique to one archetype ─────────
            ('landfall',            'Landfall'),
            ('mill',                'Mill'),
            ('death_trigger',       'Aristocrats'),
            # ── control — counterspell and card_draw before graveyard;
            # control decks run flashback/cycling spells which emit
            # MECH_graveyard but are not graveyard-strategy decks ─────
            ('counterspell',        'Control'),
            ('card_draw',           'Control'),
            # ── graveyard — after control signals ─────────────────────
            ('recursion',           'Graveyard'),
            ('graveyard_keyword',   'Graveyard'),
            ('graveyard',           'Graveyard'),
            # ── tokens ────────────────────────────────────────────────
            ('token_generation',    'Tokens'),
            ('anthem',              'Tokens'),
            # ── sacrifice ─────────────────────────────────────────────
            ('sacrifice',           'Sacrifice'),
            # ── counters ──────────────────────────────────────────────
            ('counters',            'Counters'),
            # ── remaining control signals ─────────────────────────────
            ('removal',             'Control'),
            ('control_misc',        'Control'),
            ('flash',               'Control'),
            # ── lifegain — only wins if no control/graveyard matched ──
            ('lifegain',            'Lifegain'),
            # ── ramp ──────────────────────────────────────────────────
            ('ramp',                'Ramp'),
            ('alternate_cost_ramp', 'Ramp'),
            # ── aggro ─────────────────────────────────────────────────
            ('aggro_keyword',       'Aggro'),
            ('combat',              'Aggro'),
            ('haste',               'Aggro'),
            ('trample',             'Aggro'),
            # ── weaker signals ────────────────────────────────────────
            ('flying',              'Control'),
            ('discard',             'Control'),
            # ── midrange / value ──────────────────────────────────────
            ('etb',                 'Midrange'),
            ('damage',              'Midrange'),
            ('kicker',              'Midrange'),
            ('ward',                'Midrange'),
            ('deathtouch',          'Midrange'),
            # ── last resort ───────────────────────────────────────────
            ('mana_generation',     'Ramp'),
        ]
        vote_set = {mech for mech, count in votes.items() if count > 0}
        for mech, strategy in PRIORITY:
            if mech in vote_set:
                return strategy
        return 'Midrange'

    # ------------------------------------------------------------------ #
    #  Topic naming — colour from card pool + strategy from composition   #
    # ------------------------------------------------------------------ #
    def _name_topics(self, decks: Optional[List[MetaDeck]] = None,
                     doc_term_matrix=None, card_pool: Optional[CardPool] = None):
        """
        Name topics as "<Colour> <Strategy>", e.g. "Black Control",
        "Green Landfall", "White Lifegain".

        FIX #3: sample widened to STRATEGY_SAMPLE_SIZE and deck_weights
        (the actual topic loadings) passed to _detect_strategy so that
        high-loading decks contribute proportionally more to the label.

        Skips topics that have been merged away (present in _merged_topics
        as a loser key) — they will not appear in the saved model.
        """
        used_names: Set[str] = set()

        color_names = {
            'W': 'White', 'U': 'Blue', 'B': 'Black', 'R': 'Red', 'G': 'Green',
            'WU': 'Azorius', 'WB': 'Orzhov', 'WR': 'Boros', 'WG': 'Selesnya',
            'UB': 'Dimir', 'UR': 'Izzet', 'UG': 'Simic', 'BR': 'Rakdos',
            'BG': 'Golgari', 'RG': 'Gruul',
            'WUB': 'Esper', 'WUR': 'Jeskai', 'WUG': 'Bant', 'WBR': 'Mardu',
            'WBG': 'Abzan', 'WRG': 'Naya', 'UBR': 'Grixis', 'UBG': 'Sultai',
            'URG': 'Temur', 'BRG': 'Jund',
        }

        doc_topic_matrix = None
        if decks is not None and doc_term_matrix is not None:
            doc_topic_matrix = self.model.transform(doc_term_matrix)

        for topic_idx in range(self.model.n_components):
            # FIX #4: skip merged-away (loser) topics entirely
            if topic_idx in self._merged_topics:
                continue

            individual_color_votes: Counter = Counter()
            top_decks:    List[MetaDeck] = []
            deck_weights: List[float]    = []

            if doc_topic_matrix is not None and decks is not None:
                loadings = doc_topic_matrix[:, topic_idx]
                # FIX #3: widen from top-10 to STRATEGY_SAMPLE_SIZE.
                # Also enforce a minimum loading threshold: decks where this
                # topic accounts for less than 1/n_topics of their distribution
                # are off-theme noise and must not influence the strategy label.
                # Without this, White Lifegain decks with a small secondary
                # Izzet loading pollute the Izzet topic's composition stats.
                min_loading = 1.0 / max(self.n_topics, 1)
                top_indices = loadings.argsort()[-STRATEGY_SAMPLE_SIZE:][::-1]
                for di in top_indices:
                    if di >= len(decks):
                        continue
                    weight = float(loadings[di])
                    if weight < min_loading:
                        break   # argsort is descending; all remaining are lower
                    deck = decks[di]
                    if card_pool is not None:
                        derived = card_pool.derive_deck_colors(deck.card_names)
                    else:
                        derived = deck.colors
                    for c in derived:
                        if c in 'WUBRG':
                            individual_color_votes[c] += weight
                    top_decks.append(deck)
                    deck_weights.append(weight)

            total_weight  = sum(individual_color_votes.values()) or 1.0
            top_color_pct = max(individual_color_votes.values(), default=0) / total_weight
            threshold     = 0.30 if top_color_pct >= 0.50 else 0.20
            significant   = ''.join(
                c for c in 'WUBRG'
                if individual_color_votes[c] / total_weight >= threshold
            )

            if not significant and self.model is not None:
                lda_color_votes: Counter = Counter()
                top_feat_indices = self.model.components_[topic_idx].argsort()[-40:][::-1]
                for fi in top_feat_indices:
                    feat = self.feature_names[fi]
                    if feat.startswith('COLOR_') and feat[6:] in 'WUBRG':
                        lda_color_votes[feat[6:]] += self.model.components_[topic_idx][fi]
                lda_total   = sum(lda_color_votes.values()) or 1.0
                significant = ''.join(
                    c for c in 'WUBRG'
                    if lda_color_votes[c] / lda_total >= 0.30
                )

            color_label = color_names.get(significant, significant or 'Multi') if significant else 'Multi'

            # FIX #3: pass deck_weights to _detect_strategy
            strategy = self._detect_strategy(
                top_decks, card_pool,
                topic_idx=topic_idx,
                deck_weights=deck_weights,
            ) if card_pool is not None else 'Midrange'

            base_name = f"{color_label} {strategy}"
            if base_name not in used_names:
                chosen = base_name
            else:
                suffix = 2
                while f"{base_name} {suffix}" in used_names:
                    suffix += 1
                chosen = f"{base_name} {suffix}"

            used_names.add(chosen)
            self.topic_names[f"topic_{topic_idx}"] = chosen

    # ------------------------------------------------------------------ #
    #  Main training entry point                                          #
    # ------------------------------------------------------------------ #
    def train(self, decks: List[MetaDeck], card_pool: CardPool) -> bool:
        if not decks:
            print("ERROR: No decks provided")
            return False

        if self.n_topics == 'auto':
            self.n_topics = self._auto_topic_count(len(decks))
            print(f"Auto topic count: {self.n_topics} "
                  f"(from {len(decks)} decks — change with --topics N)")
        else:
            print(f"Training LDA with {self.n_topics} topics on {len(decks)} decks...")

        deck_texts:   List[str]      = []
        deck_objects: List[MetaDeck] = []
        for deck in decks:
            text = self._extract_features(deck, card_pool)
            if len(text.split()) > 5:
                deck_texts.append(text)
                deck_objects.append(deck)

        if not deck_texts:
            print("ERROR: No usable deck text produced")
            return False

        self.vectorizer = CountVectorizer(
            max_features=6000,
            token_pattern=r'\b[A-Za-z_][A-Za-z0-9_]*\b',
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 1),
            lowercase=False,
        )
        doc_term_matrix    = self.vectorizer.fit_transform(deck_texts)
        self.feature_names = list(self.vectorizer.get_feature_names_out())
        print(f"Vocabulary size: {len(self.feature_names)}")

        print(f"Fitting LDA model ({self.n_topics} topics)...")
        self.model = self._fit_model(doc_term_matrix, self.n_topics)

        self.duplicate_pairs = self._find_duplicate_topics()
        if self.duplicate_pairs:
            print(f"  ⚠  {len(self.duplicate_pairs)} duplicate topic pair(s) detected "
                  f"— merging...")
            # FIX #4: merge duplicates now instead of just warning
            self._merged_topics = self._merge_duplicate_topics(doc_term_matrix)
        else:
            self._merged_topics = {}

        self.is_trained = True

        # ----------------------------------------------------------------
        # FIX #5: correct card_topics weights for MAX_CARD_TOKEN_COUNT cap
        #
        # Problem: model.components_[topic, feat] reflects how many times
        # feature `feat` appeared across all training documents.  Because
        # we capped name tokens at MAX_CARD_TOKEN_COUNT (= 4), a staple
        # that appears in every deck still has a large absolute column sum
        # (n_decks × 4) even though its *relative* importance per deck is
        # no higher than a card that appears in fewer decks but at full
        # count.  When we normalise raw / raw.sum() the staples still
        # dominate because their raw sum is large.
        #
        # Fix: divide each feature column by its document frequency
        # (number of docs it appears in at all, from the doc_term_matrix)
        # before computing the topic distribution.  This is conceptually
        # similar to TF-IDF reweighting applied post-hoc to the LDA
        # components matrix.
        #
        # doc_freq[i] = number of documents where feature i has count > 0
        # ----------------------------------------------------------------
        doc_freq = np.array((doc_term_matrix > 0).sum(axis=0)).flatten()
        # Avoid division by zero for features that somehow have df=0
        doc_freq = np.maximum(doc_freq, 1).astype(float)

        card_count = 0
        for idx, feature in enumerate(self.feature_names):
            if feature.startswith(('MECH_', 'TYPE_', 'COLOR_', 'CMC_')):
                continue
            norm = self._normalize_name(feature.replace('_', ' '))

            if norm in BASIC_LANDS_NORM:
                continue

            card = card_pool.cards_by_name.get(norm) or \
                   card_pool.cards_by_name.get(
                       self._normalize_name(feature.replace('_', ' ')))
            if card:
                # FIX #5: divide raw component weight by document frequency
                # before normalising so that ubiquitous staples don't absorb
                # disproportionate topic probability mass.
                raw   = self.model.components_[:, idx].copy() / doc_freq[idx]

                # FIX #4: zero out merged (loser) topic slots so their
                # probability mass doesn't pollute the surviving topic's
                # card distribution
                for loser_idx in self._merged_topics:
                    raw[loser_idx] = 0.0

                total = raw.sum() + 1e-10
                dist  = raw / total

                # Only store non-loser topics in the output dict
                self.card_topics[norm] = {
                    f"topic_{i}": float(dist[i])
                    for i in range(self.n_topics)
                    if i not in self._merged_topics
                }
                card_count += 1

        print(f"Mapped {card_count} cards to topics")
        self._name_topics(decks=deck_objects, doc_term_matrix=doc_term_matrix,
                          card_pool=card_pool)
        return True

    # ------------------------------------------------------------------ #
    #  Reporting                                                          #
    # ------------------------------------------------------------------ #
    def print_report(self):
        print("\n" + "=" * 60)
        print("LDA TRAINING REPORT")
        print("=" * 60)

        active_topics = [
            (tid, tname) for tid, tname in self.topic_names.items()
            if int(tid.split('_')[1]) not in self._merged_topics
        ]
        print(f"Topics: {len(active_topics)} active "
              f"({len(self._merged_topics)} merged)  |  "
              f"Cards mapped: {len(self.card_topics)}")

        for tid, tname in active_topics:
            t_idx       = int(tid.split('_')[1])
            top_indices = self.model.components_[t_idx].argsort()[-20:][::-1]

            top_cards  = []
            top_mechs  = []
            top_types  = []
            top_colors = []
            seen_m: set = set()

            for i in top_indices:
                feat = self.feature_names[i]
                if not feat.startswith(('MECH_', 'TYPE_', 'COLOR_', 'CMC_')):
                    norm = self._normalize_name(feat.replace('_', ' '))
                    if norm not in BASIC_LANDS_NORM:
                        top_cards.append(feat.replace('_', ' ').title())
                elif feat.startswith('MECH_') and feat[5:] not in seen_m:
                    top_mechs.append(feat[5:])
                    seen_m.add(feat[5:])
                elif feat.startswith('TYPE_'):
                    top_types.append(feat[5:])
                elif feat.startswith('COLOR_'):
                    top_colors.append(feat[6:])

            print(f"\n{tname}:")
            if top_cards:
                print(f"  Cards: {', '.join(top_cards[:8])}")
            if top_mechs:
                print(f"  Mechanics: {', '.join(top_mechs[:5])}")
            if top_types:
                print(f"  Types: {', '.join(dict.fromkeys(top_types))}")
            if top_colors:
                print(f"  Colors: {''.join(dict.fromkeys(top_colors))}")

        if self._merged_topics:
            print("\nMerged topics (removed as duplicates):")
            for loser, survivor in self._merged_topics.items():
                survivor_name = self.topic_names.get(f"topic_{survivor}", f"topic_{survivor}")
                print(f"  topic_{loser} → {survivor_name}")

        print("\n" + "-" * 60)
        print("Topic similarity (Jensen-Shannon divergence — lower = more similar):")
        names = [self.topic_names[f"topic_{i}"]
                 for i in range(self.model.n_components)
                 if i not in self._merged_topics]
        active_indices = [i for i in range(self.model.n_components)
                          if i not in self._merged_topics]
        if names:
            col_w = max(len(nm) for nm in names)
            print(f"{'':>{col_w}}" + "".join(f"  {nm[:10]:>10}" for nm in names))
            for ii, i in enumerate(active_indices):
                row = f"{names[ii]:>{col_w}}"
                for j in active_indices:
                    if i == j:
                        row += f"  {'---':>10}"
                    else:
                        jsd  = self._jsd(self.model.components_[i],
                                         self.model.components_[j])
                        flag = "⚠" if jsd < JSD_DUPLICATE_THRESHOLD else " "
                        row += f"  {flag}{jsd:>6.3f}   "
                print(row)

        print(f"\n💡 Tip: if topics still overlap, try --topics "
              f"{max(2, len(active_topics) - 1)}")

    # ------------------------------------------------------------------ #
    #  Save                                                               #
    # ------------------------------------------------------------------ #
    def save(self, filepath: str = LDA_MODEL_PATH):
        # FIX #4: n_topics in the saved model reflects the post-merge count
        active_count = self.model.n_components - len(self._merged_topics)
        data = {
            'card_topics':     self.card_topics,
            'topic_names':     self.topic_names,    # only surviving topics present
            'n_topics':        active_count,         # FIX #4: post-merge count
            'n_topics_raw':    self.model.n_components,
            'is_trained':      self.is_trained,
            'merged_topics':   {str(k): v for k, v in self._merged_topics.items()},
            'duplicate_pairs': [
                {'i': i, 'j': j, 'jsd': jsd}
                for i, j, jsd in self.duplicate_pairs
            ],
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"\n✓ Model saved to {filepath}")
        print(f"  {len(self.card_topics)} card-topic mappings")
        print(f"  {len(self.topic_names)} named archetypes ({active_count} active)")


# ---------- MAIN ----------
def main():
    parser = argparse.ArgumentParser(
        description='Train LDA model for MTG deck archetypes')
    parser.add_argument('--decks',     type=int, default=None,
                        help='Max decks to use (default: all)')
    parser.add_argument('--topics',    type=str, default=str(LDA_TOPICS),
                        help='Number of topics, or "auto" (default: auto)')
    parser.add_argument('--all-decks', action='store_true',
                        help='Use all decks (not just is_meta=1)')
    parser.add_argument('--deck-dir',  type=str, default=str(FORGE_DECK_DIR),
                        help='Deck directory path')
    parser.add_argument('--output',    type=str, default=LDA_MODEL_PATH,
                        help=f'Output JSON path (default: {LDA_MODEL_PATH})')
    args = parser.parse_args()

    if args.topics.lower() == 'auto':
        n_topics = 'auto'
    else:
        try:
            n_topics = int(args.topics)
            if n_topics < 2:
                print("ERROR: --topics must be at least 2")
                return
        except ValueError:
            print(f"ERROR: invalid --topics value '{args.topics}'")
            return

    print("=" * 60)
    print("MTG LDA Model Trainer")
    print("=" * 60)

    loader = DeckLoader(db_path=DB_PATH, deck_dir=Path(args.deck_dir))
    decks  = loader.load_decks(max_decks=args.decks, meta_only=not args.all_decks)

    if len(decks) < 5:
        print("ERROR: Not enough decks (need at least 5)")
        return

    print("\nLoading card database...")
    with CardPool() as pool:
        if not pool.cards_by_name:
            print("ERROR: No cards found in database")
            return
        print(f"Loaded {len(pool.cards_by_name)} unique card entries")

        trainer = LDATrainer(n_topics=n_topics)
        if trainer.train(decks, pool):
            trainer.print_report()
            trainer.save(filepath=args.output)
            print("\n✓ Training complete!")
            print(f"Next step: run the deckbuilder pointing at {args.output}")
        else:
            print("ERROR: Training failed")


if __name__ == "__main__":
    main()
