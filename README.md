# Retail Intelligence Dashboard with Explainable AI

This workspace is being scaffolded for a Streamlit MVP focused on retail/superstore-style transactional datasets.

## Planned MVP Structure

- `app.py` - Streamlit entrypoint and page flow
- `requirements.txt` - Python dependencies
- `.gitignore` - Local secrets and environment files
- `utils/` - Modular helpers for loading, validation, mapping, analytics, explainability, and RAG

### Expected App Flow

1. Landing page and file upload
2. LLM-assisted column mapping with clarification loop
3. Adaptive dashboard with charts and KPIs
4. Chat with grounded answers from computed findings

### Notes

- This MVP is for order-level retail data such as Superstore-style CSV/XLSX files.
- Gemini API access is read from `.env` through `python-dotenv` (`GEMINI_API_KEY` or `GOOGLE_API_KEY`), not hardcoded in the app.
- Module 10, delivery estimation, auth, persistence, and deployment are intentionally out of scope for this phase.

### Local Setup

1. Create a `.env` file at the project root.
2. Add `GEMINI_API_KEY=your_key_here`.
3. Run `streamlit run app.py`.
