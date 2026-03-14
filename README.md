🃏 MTG Arena Genetic Deckbuilder with LDA
An automated Magic: The Gathering deck-building pipeline that combines machine learning (Latent Dirichlet Allocation) with a genetic algorithm to evolve optimized decks from your personal MTGA card collection.
The system scrapes current metagame data, learns archetype patterns from real meta decks, and then evolves a competitive 60-card deck using only cards you actually own.
---
🧠 How It Works
```
Your MTGA Collection
        │
        ▼
\[1] import\_card\_pool.py   ──► mtg\_cards.db       (your owned cards)
        
\[2] mtg\_scraper.py        ──► decks\_cache.db      (meta deck data)
        
\[3] train\_lda.py          ──► lda\_model.json      (archetype model)
        
\[4] mtg\_genetic\_deckbuilder.py  ──► final\_ga\_deck.dck  (your optimized deck)
```
Import your card pool — Parse your AetherHub collection export and enrich each card with data from the Scryfall API, storing everything in a local SQLite database.
Scrape the metagame — Pull Standard meta decks from AetherHub and MTGGoldfish into a second database, enriched with Scryfall card data.
Train the LDA model — Use Latent Dirichlet Allocation to discover deck archetypes (topics) across all scraped meta decks, mapping every card to its archetype distribution.
Evolve your deck — Run a DEAP-powered genetic algorithm that initializes individuals using LDA archetype seeding, evolves them over generations using crossover and mutation, and exports the best result in Forge `.dck` format.
---
📋 Requirements
Python 3.8+
MTG Forge (for live playtesting mode)
An AetherHub collection export (UTF-16 `.txt`)
Python Dependencies
```bash
pip install requests beautifulsoup4 scikit-learn numpy deap
```
---
🚀 Quickstart
Step 1 — Import Your Card Pool
Export your collection from AetherHub, then run:
```bash
python3 import\_card\_pool.py my\_collection.txt
```
Options:
Flag	Description
`--format standard`	Filter to Standard-legal cards only (default)
`--format none`	Import all cards regardless of legality
`--skip-scryfall`	Use cached card data only, skip API calls
`--force-refresh`	Re-fetch all card data from Scryfall
This creates `mtg\_cards.db` with your card pool, mana production data, and inferred card roles (threat, removal, ramp, etc.).
---
Step 2 — Scrape Meta Decks
```bash
python3 mtg\_scraper.py
```
Scrapes both AetherHub (Standard BO1) and MTGGoldfish (Standard) by default.
Options:
Flag	Description
`--source all`	Scrape both sources (default)
`--source aetherhub`	AetherHub only
`--source mtggoldfish`	MTGGoldfish only
`--meta 75`	Number of AetherHub meta decks to fetch (default: 50)
`--pages 6`	AetherHub user-deck pages (default: 4)
`--max-decks 80`	MTGGoldfish deck cap (default: 60)
`--backfill-only`	Re-enrich card data without re-scraping
`--backfill-only --fix-colors`	Re-fetch cards with missing color data
`--force`	Clear all decks and re-scrape from scratch
This creates `decks\_cache.db` with deck lists and full Scryfall card data.
---
Step 3 — Train the LDA Model
```bash
python3 train\_lda.py --all-decks
```
Options:
Flag	Description
`--all-decks`	Use all decks in the DB (recommended)
`--topics 7`	Set a fixed number of archetypes
`--topics auto`	Auto-select optimal topic count (default)
`--decks 200`	Cap the number of decks used for training
`--output lda\_model.json`	Output path for the trained model
The trainer will print a report showing discovered archetypes (e.g., "Aggro", "White Lifegain", "Sacrifice"), their top cards, mechanics, colors, and a Jensen-Shannon divergence matrix to detect overlapping topics. Duplicate topics are automatically merged.
Output: `lda\_model.json`
---
Step 4 — Build Your Deck
```bash
python3 mtg\_genetic\_deckbuilder.py --colors WB --generations 30 --population 50
```
Required:
Flag	Description
`--colors` / `-c`	Color identity for the deck (e.g., `WB`, `RG`, `WUBR`)
Options:
Flag	Description
`--generations` / `-g`	Number of GA generations (default: 20)
`--population` / `-p`	Population size (default: 30)
`--archetype` / `-a`	Force a specific LDA archetype (e.g., `"White Lifegain"`)

`--mode sim`	Use simulation scoring (no Forge required)
`--mode live`	Use Forge for live playtesting (default)
`--games`	Games per matchup in live mode (default: 5)
`--resume` / `-r`	Resume from the latest checkpoint
`--fresh` / `-f`	Force a fresh run, ignoring checkpoints
`--debug`	Enable verbose debug logging
Outputs:
`final\_ga\_deck.dck` — Forge-ready deck file
`final\_ga\_deck.txt` — Human-readable deck list with archetype and fitness score
---
📁 File & Database Overview
File	Created By	Description
`mtg\_cards.db`	`import\_card\_pool.py`	Your owned card pool with Scryfall data, roles, and mana production
`scryfall\_cache.json`	`import\_card\_pool.py`	Local cache of Scryfall API responses
`decks\_cache.db`	`mtg\_scraper.py`	Meta deck lists with full card data
`lda\_model.json`	`train\_lda.py`	Trained topic model mapping cards to archetypes
`checkpoints/`	`mtg\_genetic\_deckbuilder.py`	Resumable GA state (JSON, last 3 kept)
`final\_ga\_deck.dck`	`mtg\_genetic\_deckbuilder.py`	Forge-format output deck
`final\_ga\_deck.txt`	`mtg\_genetic\_deckbuilder.py`	Text decklist with metadata
---
🔬 Technical Details
LDA Archetype Discovery
The LDA model treats each deck as a "document" and each card as a "word." Training builds a topic model where each topic represents an emergent archetype pattern. Cards are encoded not just by name but also by mechanic tokens (`MECH\_haste`, `MECH\_sacrifice`), type tokens (`TYPE\_Creature`), color tokens (`COLOR\_W`), and CMC buckets (`CMC\_2`). This lets the model capture deck strategies that share mechanics even when specific card names differ.
Duplicate topics are detected using Jensen-Shannon divergence and automatically collapsed before saving, keeping the model clean.
Genetic Algorithm
Individual — A 60-card deck represented as a list of card IDs
Initialization — Individuals are seeded from LDA topic distributions to start near coherent archetypes rather than random card combinations
Crossover — Deck segments are swapped between parents while preserving card count constraints
Mutation — Cards are replaced with alternatives drawn from the same LDA topic, preserving archetype coherence
Fitness — Scored by win rate against meta decks (live or simulated), archetype coherence entropy, mana curve shape, and color consistency
Checkpointing — State is saved every generation in JSON format; press `Ctrl+C` for a graceful emergency save
Color & Format Integrity
The importer validates card colors against their mana costs, handles double-faced cards (DFCs), and auto-fixes known mismatches. Only Standard-legal cards are imported by default. The scraper also backfills and repairs cards with missing color data.
---
💡 Tips
Run `train\_lda.py` again after adding more decks to `decks\_cache.db` — more data means better archetypes.
If topics overlap in the JSD matrix report, re-run with `--topics N` where N is one less than the current active count.
Use `--archetype "Aggro"` (or whichever archetype the LDA report names) to constrain the GA toward a specific strategy.
In `sim` mode no Forge install is needed, making it fast for experimentation; switch to `live` for more accurate win-rate evaluation.
Interrupt a long run at any time with `Ctrl+C` and resume later with `--resume`.
---
📜 License
MIT License — use freely, brew responsibly.
