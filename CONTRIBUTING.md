# Contributing

## Setup
git clone https://github.com/seya-makin/cobblestone-power
cd cobblestone-power
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add ENTSOE_API_KEY and GEMINI_API_KEY to .env

## Running Tests
python -m pytest tests/ -v

## Running the Pipeline
python run_pipeline.py --mode full

## Code Style
- Type hints on all functions
- Docstrings on all classes and methods
- All file paths via pathlib.Path
- All secrets via environment variables only

## Adding a New Data Source
1. Create ingestion method in src/ingestion.py
2. Add cleaning logic in src/cleaning.py
3. Document endpoint in docs/api_endpoints.md
4. Add unit test in tests/
