# Copilot / AI Agent Instructions — Camera Product Comparison

This guide enables AI coding agents to be productive in this codebase immediately. It summarizes architecture, workflows, conventions, and integration points specific to this project.

## Architecture Overview
- **Frontend/UI**: [app.py](app.py) (Streamlit) — manages user interaction, session state (`agent`, `messages`), and chart display.
- **Agent Layer**: [main_agent.py](main_agent.py) — `CameraAgent` class parses user intent (via LLM), queries the database, interprets results, and orchestrates chart generation.
- **Database**: PostgreSQL accessed via SQLAlchemy ([models.py](models.py)). Connection URI in [config.py](config.py) as `DB_URI`.
- **ETL/Seed Data**: [db_loader.py](db_loader.py) loads [data/camera_data_clean4.csv](data/camera_data_clean4.csv) (encoding `gbk`) into the `cameras` table. Missing prices are simulated.
- **Visualization**: [visualizer.py](visualizer.py) (matplotlib) generates PNGs in `charts/` and returns file paths for Streamlit display.
- **Scraping/Data Sources**: [重新爬取数据/scrape.js](重新爬取数据/scrape.js) scrapes DXOMark-style data into `scraped/*.txt`; large CSV/JSON files are under `data/`.

## Key Developer Workflows
- **Run UI locally**: `streamlit run app.py`
- **Load/refresh DB**: `python db_loader.py` (clears and repopulates `cameras` table; simulates `Price` if missing)
- **DB setup**: Create PostgreSQL DB, set `DB_URI` in [config.py](config.py) (see example in file)

## Integration & Configuration
- [config.py](config.py):
  - `DB_URI`: SQLAlchemy URI (e.g., `postgresql://user:pw@localhost:5432/camera_db`)
  - `DASHSCOPE_API_KEY`, `LLM_MODEL`: LLM client uses OpenAI-compatible API (Dashscope)
- LLM contract: `_parse_intent` in [main_agent.py](main_agent.py) expects strict JSON: `{ "max_price": number, "sort_field": "fieldName", "summary": "text" }`

## Project-Specific Conventions
- **CSV encoding**: Always use `encoding='gbk'` for CSVs in [db_loader.py](db_loader.py)
- **Null handling**: Use `df = df.where(pd.notnull(df), None)` before DB insert
- **Price simulation**: If missing, `db_loader.py` generates `simulated_price = random.randint(30, 250) * 100`
- **Visualizer**: 
  - `draw_radar(camera)`: expects fields like `Portability_Score`, `LowLight_Score`, `Video_Score`, `Max_ISO`
  - `draw_comparison(cameras, field_name)`: `field_name` must exist on `Camera`
  - `draw_price_performance(cameras, all_cameras)`: computes `(LowLight_Score + Video_Score)/2`
- **DB schema**: [models.py](models.py) `Camera` fields must match ETL columns
- **Query limits**: [main_agent.py](main_agent.py) limits results to 3 (recommendations) and 100 (scatter)

## Where to Change Behavior
- **Intent parsing/LLM**: `_parse_intent` in [main_agent.py](main_agent.py)
- **Agent logic**: `handle_chat`, `_get_expert_reply` in [main_agent.py](main_agent.py)
- **Schema/ETL**: [models.py](models.py), [db_loader.py](db_loader.py)
- **Visualization**: [visualizer.py](visualizer.py)
- **UI/session**: [app.py](app.py)

## Dependencies & Setup
- No `requirements.txt` — install: `streamlit`, `sqlalchemy`, `psycopg2-binary`, `pandas`, `matplotlib`, `numpy`, `openai`
- LLM: Uses `openai.OpenAI(..., base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")`; set `DASHSCOPE_API_KEY` in [config.py](config.py)

## Example Local Setup
1. Create DB and user
2. `pip install streamlit sqlalchemy psycopg2-binary pandas matplotlib numpy openai`
3. `python db_loader.py`
4. `streamlit run app.py`

## Security Note
- [config.py](config.py) stores secrets in plaintext. Prefer environment variables or a `.env` file (not present).

---
If you need: a `requirements.txt`, `.env` support, or a `CONTRIBUTING.md`, request it. For unclear or missing sections, ask for clarification or more details.