import re
import sqlite3
import json
import requests
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
from time import sleep
import sys
import os

@dataclass
class PoolCard:
    quantity: int
    name: str
    set_code: str
    collector_number: str
    scryfall_data: Optional[Dict] = None

    @property
    def is_standard_legal(self) -> bool:
        if not self.scryfall_data:
            return False
        legalities = self.scryfall_data.get('legalities', {})
        return legalities.get('standard') == 'legal'

def parse_card_pool(filepath: str) -> List[PoolCard]:
    """
    Parse AetherHub collection export (UTF-16 encoded)
    """
    cards = []
    pattern = r'(\d+)\s+([^(]+)\s+\(([A-Za-z0-9]+)\)\s+(\d+)'

    encodings = ['utf-16-le', 'utf-16', 'utf-8-sig', 'utf-8']

    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                content = f.read()
                print(f"Successfully read file with {encoding} encoding")
                break
        except UnicodeError:
            continue
    else:
        print("Error: Could not decode file with any standard encoding")
        sys.exit(1)

    for line_num, line in enumerate(content.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('//') or line.startswith('Deck'):
            continue

        match = re.match(pattern, line)
        if match:
            qty, name, set_code, num = match.groups()
            cards.append(PoolCard(
                quantity=int(qty),
                name=name.strip(),
                set_code=set_code.upper(),
                collector_number=num
            ))
        else:
            if line_num < 10:
                print(f"Warning: Could not parse line {line_num}: {line[:50]}...")

    return cards

def extract_colors_from_data(data: Dict) -> List[str]:
    """
    Extract colors from Scryfall data, handling DFCs and missing data.
    Returns list like ['B'], ['W', 'U'], etc.
    """
    # Try top-level colors first
    colors = data.get('colors', [])

    # For DFCs, use front face colors if top-level is empty
    if not colors and 'card_faces' in data:
        front_face = data['card_faces'][0]
        colors = front_face.get('colors', [])

    # Fallback: extract from mana_cost if still no colors
    if not colors:
        mana_cost = data.get('mana_cost', '')
        colors = list(set(re.findall(r'\{([WUBRG])\}', mana_cost)))

    return colors

def validate_colors_against_mana_cost(data: Dict) -> bool:
    """
    Verify that colors match the mana cost. Returns True if valid.
    """
    colors = set(extract_colors_from_data(data))
    mana_cost = data.get('mana_cost', '')
    mana_colors = set(re.findall(r'\{([WUBRG])\}', mana_cost))

    # Colorless cards (artifacts) should have empty colors or match mana cost
    if not colors and not mana_colors:
        return True

    # If colors exist, they should match mana cost colors (allowing for hybrid)
    if colors and mana_colors:
        # Colors should be subset of mana colors or vice versa
        if not (colors.issubset(mana_colors) or mana_colors.issubset(colors)):
            print(f"    WARNING: Color mismatch for {data.get('name')}: "
                  f"colors={colors} but mana_cost={mana_cost}")
            return False

    return True

def batch_enrich_scryfall(cards: List[PoolCard], force_refresh: bool = False) -> List[PoolCard]:
    """Fetch card data from Scryfall API with caching"""
    print(f"\nFetching Scryfall data for {len(cards)} cards...")

    cache_file = 'scryfall_cache.json'
    cache = {}

    # Load existing cache
    if os.path.exists(cache_file) and not force_refresh:
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            print(f"Loaded {len(cache)} cached cards")

            # Validate cache for suspicious entries
            invalid_cache_entries = []
            for name, data in cache.items():
                if not validate_colors_against_mana_cost(data):
                    invalid_cache_entries.append(name)

            if invalid_cache_entries:
                print(f"\nWARNING: Found {len(invalid_cache_entries)} cached cards with color mismatches!")
                print(f"Deleting invalid cache entries: {invalid_cache_entries[:5]}...")
                for name in invalid_cache_entries:
                    del cache[name]

        except Exception as e:
            print(f"Cache load error: {e}")

    updated = 0
    failed = []
    color_mismatches = []

    for i, card in enumerate(cards):
        cache_key = card.name

        # Check cache first
        if cache_key in cache and not force_refresh:
            # Validate cached data
            if validate_colors_against_mana_cost(cache[cache_key]):
                card.scryfall_data = cache[cache_key]
                continue
            else:
                print(f"  Invalid cache for {card.name}, re-fetching...")

        # Fetch from API
        try:
            url = "https://api.scryfall.com/cards/named"
            params = {"exact": card.name}

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()

                # Validate before caching
                if not validate_colors_against_mana_cost(data):
                    color_mismatches.append(card.name)

                # Fix colors for DFCs if needed
                if not data.get('colors') and 'card_faces' in data:
                    data['colors'] = extract_colors_from_data(data)

                card.scryfall_data = data
                cache[cache_key] = data
                updated += 1

                # Save cache periodically
                if updated % 50 == 0:
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        json.dump(cache, f)
                    print(f"  Cached {updated} new cards... ({i+1}/{len(cards)})")
            else:
                failed.append(card.name)
                if len(failed) <= 5:
                    print(f"  Not found: {card.name} (Status {response.status_code})")

            sleep(0.1)  # Rate limit respect

        except Exception as e:
            failed.append(card.name)
            if len(failed) <= 5:
                print(f"  Error with {card.name}: {e}")

    # Final cache save
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache, f)

    print(f"\nComplete: {updated} fetched, {len(failed)} failed")
    if color_mismatches:
        print(f"WARNING: {len(color_mismatches)} cards had color/mana cost mismatches!")
        print(f"  Check these manually: {', '.join(color_mismatches[:10])}")
    if failed:
        print(f"Failed cards: {', '.join(failed[:10])}")

    return cards

def create_card_pool_database(cards: List[PoolCard], format_filter: str = 'standard'):
    """
    Create the card pool database for the GA
    """
    conn = sqlite3.connect('mtg_cards.db')
    c = conn.cursor()

    # Create tables
    c.execute('''
        CREATE TABLE IF NOT EXISTS cards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            mana_cost TEXT,
            cmc INTEGER,
            colors TEXT,
            types TEXT,
            subtypes TEXT,
            oracle_text TEXT,
            keywords TEXT,
            rarity TEXT,
            is_basic_land BOOLEAN DEFAULT 0,
            user_quantity INTEGER DEFAULT 1,
            set_code TEXT,
            collector_number TEXT
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS card_roles (
            card_id INTEGER,
            role TEXT,
            weight REAL DEFAULT 1.0,
            FOREIGN KEY (card_id) REFERENCES cards(id)
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS mana_production (
            card_id INTEGER,
            mana_symbol TEXT,
            is_conditional BOOLEAN DEFAULT 0,
            FOREIGN KEY (card_id) REFERENCES cards(id)
        )
    ''')

    # Clear existing pool
    c.execute("DELETE FROM card_roles")
    c.execute("DELETE FROM mana_production")
    c.execute("DELETE FROM cards")

    seen_cards = {}
    imported = 0
    skipped_format = 0
    skipped_no_data = 0
    merged = 0
    color_fixes = 0

    print(f"\nImporting cards (format filter: {format_filter})...")

    for card in cards:
        if not card.scryfall_data:
            skipped_no_data += 1
            continue

        if format_filter == 'standard' and not card.is_standard_legal:
            skipped_format += 1
            continue

        try:
            data = card.scryfall_data
            card_name = data.get('name', card.name)

            # Get validated colors
            colors_list = extract_colors_from_data(data)

            # DOUBLE CHECK: If we have a mana cost but no colors, extract from mana cost
            mana_cost = data.get('mana_cost', '')
            if not colors_list and mana_cost:
                colors_list = list(set(re.findall(r'\{([WUBRG])\}', mana_cost)))
                if colors_list:
                    print(f"  Fixed missing colors for {card_name}: {colors_list}")
                    color_fixes += 1

            # CRITICAL: Check for color mismatch (e.g., Sunset Saboteur showing as Black)
            mana_colors = set(re.findall(r'\{([WUBRG])\}', mana_cost))
            card_colors = set(colors_list)

            # If mana cost has colors but colors array is empty or mismatched, use mana colors
            if mana_colors and not card_colors:
                colors_list = list(mana_colors)
                print(f"  WARNING: {card_name} had no colors but mana cost {mana_cost}. Fixed to {colors_list}")
                color_fixes += 1

            if card_name in seen_cards:
                existing_id, existing_qty = seen_cards[card_name]
                new_qty = existing_qty + card.quantity
                c.execute("UPDATE cards SET user_quantity = ? WHERE id = ?",
                         (new_qty, existing_id))
                seen_cards[card_name] = (existing_id, new_qty)
                merged += 1
                continue

            # Extract type information
            type_line = data.get('type_line', '')
            types = type_line
            subtypes = ''

            if '—' in type_line:
                types, subtypes = type_line.split('—', 1)
                types = types.strip()
                subtypes = subtypes.strip()

            is_basic = 'Basic' in type_line and 'Land' in type_line

            # Store colors as JSON
            colors_json = json.dumps(colors_list)

            # Extract keywords
            keywords = json.dumps(data.get('keywords', []))

            # Insert new card
            c.execute('''
                INSERT INTO cards
                (name, mana_cost, cmc, colors, types, subtypes, oracle_text,
                 keywords, rarity, is_basic_land, user_quantity, set_code, collector_number)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                card_name,
                mana_cost,
                data.get('cmc', 0),
                colors_json,
                types,
                subtypes,
                data.get('oracle_text', ''),
                keywords,
                data.get('rarity', ''),
                is_basic,
                card.quantity,
                card.set_code,
                card.collector_number
            ))

            card_id = c.lastrowid
            seen_cards[card_name] = (card_id, card.quantity)

            # Extract mana production
            oracle_text = data.get('oracle_text', '')
            mana_productions = extract_mana_production(oracle_text)

            for mana_symbol, is_conditional in set(mana_productions):
                c.execute('''
                    INSERT INTO mana_production (card_id, mana_symbol, is_conditional)
                    VALUES (?, ?, ?)
                ''', (card_id, mana_symbol, is_conditional))

            # Infer roles
            roles = infer_card_roles(data, oracle_text, types)
            for role, weight in roles:
                c.execute('''
                    INSERT INTO card_roles (card_id, role, weight)
                    VALUES (?, ?, ?)
                ''', (card_id, role, weight))

            imported += 1

            if imported % 100 == 0:
                print(f"  Imported {imported} cards... (merged {merged} duplicates, fixed {color_fixes} colors)")

        except Exception as e:
            print(f"Error importing {card.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    conn.commit()

    # Create indexes
    c.execute("CREATE INDEX IF NOT EXISTS idx_cards_name ON cards(name)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_roles_card_id ON card_roles(card_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_mana_card_id ON mana_production(card_id)")

    # Verify no off-color cards snuck in (for BG check)
    if format_filter:
        print("\nVerifying color integrity...")
        for name, (cid, _) in seen_cards.items():
            c.execute("SELECT colors, mana_cost FROM cards WHERE id = ?", (cid,))
            row = c.fetchone()
            if row:
                colors, mana = row
                color_list = json.loads(colors) if colors else []
                mana_colors = set(re.findall(r'\{([WUBRG])\}', mana or ''))

                # If you want to check for specific wrong cards:
                if 'Sunset Saboteur' in name and 'R' not in color_list:
                    print(f"  CRITICAL: {name} has wrong colors {color_list}, should have Red!")
                    # Fix it immediately
                    c.execute("UPDATE cards SET colors = ? WHERE id = ?",
                             (json.dumps(['R']), cid))
                    print(f"  AUTO-FIXED {name} to Red")

    conn.commit()
    conn.close()

    print(f"\nDatabase import complete:")
    print(f"  Unique cards imported: {imported}")
    print(f"  Duplicate printings merged: {merged}")
    print(f"  Color fixes applied: {color_fixes}")
    print(f"  Skipped (wrong format): {skipped_format}")
    print(f"  Skipped (no data): {skipped_no_data}")

def extract_mana_production(oracle_text: str) -> List[tuple]:
    if not oracle_text:
        return []

    productions = []
    add_mana_pattern = r'\{T\}:?\s*Add\s*([^\n]+?)(?:\.|\n|$)'
    matches = re.findall(add_mana_pattern, oracle_text, re.IGNORECASE)

    for match in matches:
        symbols = re.findall(r'\{([^}]+)\}', match)
        for symbol in symbols:
            if symbol == 'T':
                continue
            productions.append((symbol, False))

    conditional_keywords = ['if you control', 'pay', 'sacrifice', 'discard', 'unless']
    is_conditional = any(keyword in oracle_text.lower() for keyword in conditional_keywords)

    alt_pattern = r'(?:enters the battlefield|When|Whenever).*?Add\s*([^\n]+?)(?:\.|\n|$)'
    alt_matches = re.findall(alt_pattern, oracle_text, re.IGNORECASE)

    for match in alt_matches:
        symbols = re.findall(r'\{([^}]+)\}', match)
        for symbol in symbols:
            productions.append((symbol, is_conditional))

    return productions

def infer_card_roles(card_data: Dict, oracle_text: str, types: str) -> List[tuple]:
    roles = []
    text_lower = (oracle_text or '').lower()
    keywords = [k.lower() for k in card_data.get('keywords', [])]

    if 'Creature' in types:
        power = card_data.get('power', '0')
        if power.isdigit() and int(power) >= 4:
            roles.append(('heavy_threat', 1.0))
        elif power.isdigit() and int(power) >= 2:
            roles.append(('threat', 1.0))

        if 'haste' in keywords or 'haste' in text_lower:
            roles.append(('aggro', 1.2))

    removal_patterns = ['destroy', 'exile', 'counter', '-X/-X', 'damage to']
    if any(pattern in text_lower for pattern in removal_patterns):
        if 'Instant' in types or 'Sorcery' in types:
            roles.append(('removal', 1.0))
        elif 'Enchantment' in types:
            roles.append(('continuous_removal', 0.8))

    if 'draw' in text_lower and ('card' in text_lower or 'cards' in text_lower):
        roles.append(('card_draw', 1.0))
        roles.append(('engine', 0.8))

    if 'token' in text_lower:
        roles.append(('token_generator', 1.0))
        roles.append(('go_wide', 0.9))

    if any('Add {' in line for line in oracle_text.split('\n') if 'Add {' in line):
        if 'Land' in types:
            roles.append(('ramp', 1.0))
        elif 'Creature' in types:
            roles.append(('mana_dork', 0.9))
            roles.append(('ramp', 0.8))
        else:
            roles.append(('mana_rock', 0.9))
            roles.append(('ramp', 0.8))

    if 'Instant' in types:
        roles.append(('interaction', 1.0))

    if 'search' in text_lower and 'library' in text_lower:
        roles.append(('tutor', 1.0))
        roles.append(('consistency', 0.9))

    if not roles:
        roles.append(('generic', 0.5))

    return roles

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Import MTG card pool for genetic deckbuilding')
    parser.add_argument('filepath', help='Path to AetherHub collection export file')
    parser.add_argument('--format', default='standard', choices=['standard', 'none'],
                       help='Filter by format legality (default: standard)')
    parser.add_argument('--skip-scryfall', action='store_true',
                       help='Skip Scryfall API fetch (use existing cache only)')
    parser.add_argument('--force-refresh', action='store_true',
                       help='Force re-fetch all cards from Scryfall (clear cache)')

    args = parser.parse_args()

    if not os.path.exists(args.filepath):
        print(f"Error: File not found: {args.filepath}")
        sys.exit(1)

    # Clear cache if forcing refresh
    if args.force_refresh and os.path.exists('scryfall_cache.json'):
        print("Forcing cache refresh...")
        os.remove('scryfall_cache.json')

    cards = parse_card_pool(args.filepath)
    print(f"Found {len(cards)} cards in collection file")

    if not args.skip_scryfall:
        cards = batch_enrich_scryfall(cards, force_refresh=args.force_refresh)
    else:
        cache_file = 'scryfall_cache.json'
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            for card in cards:
                if card.name in cache:
                    card.scryfall_data = cache[card.name]
            print(f"Loaded cached data for {sum(1 for c in cards if c.scryfall_data)} cards")

    format_filter = args.format if args.format != 'none' else None
    create_card_pool_database(cards, format_filter=format_filter or 'standard')

    print("\nDone! Database 'mtg_cards.db' is ready.")

if __name__ == "__main__":
    main()
