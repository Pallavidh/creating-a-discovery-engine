# Photo Recall Discovery Engine

Finds out why people fail to retrieve a photo they remember but can't precisely describe, using public Google Photos feedback.

**Pipeline:** collect posts → filter to photo-finding posts → extract structured tags with Claude → compare problems → answer research questions with cited evidence.

## Files

| File | What it does |
|---|---|
| `collect_data.ipynb` | Colab notebook. Pulls Play Store, App Store and Reddit data into `corpus.csv`. |
| `engine.py` | Tagging schema, Claude extraction, aggregation, opportunity scoring, Q&A. |
| `app.py` | The public Streamlit app. |
| `data/tagged.csv` | The tagged corpus every visitor sees (you add this). |

## Setup (about 45 minutes)

**1. Collect data**
Open `collect_data.ipynb` in Google Colab (File > Upload notebook) and run all cells. It downloads `corpus.csv`.

**2. Put the code on GitHub**
Create a public repo and upload everything in this folder except `corpus.csv`. Keep the `.streamlit` folder, but never upload a real `secrets.toml`.

**3. Deploy**
Go to share.streamlit.io, sign in with GitHub, choose **Create app**, pick the repo and set the main file to `app.py`.
Under **Advanced settings > Secrets**, paste:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
ADMIN_PASSWORD = "your-password"
```

**4. Tag the full corpus**
Open the app, enter your password in the sidebar, go to **Run the pipeline**, upload `corpus.csv` and run it. Download `tagged.csv` when it finishes.

Alternative for very large files, run locally or in Colab:
```bash
pip install -r requirements.txt
ANTHROPIC_API_KEY=sk-ant-... python engine.py corpus.csv data/tagged.csv
```

**5. Make it permanent**
Upload `tagged.csv` to the repo as `data/tagged.csv`. The app redeploys, and every visitor now sees your findings.

**6. Test like an evaluator**
Open the app link in an incognito window. Check that the dashboard loads, one question in **Ask the data** answers, and **Tag a single post** works.

## Cost and limits

- Tagging uses Claude Haiku 4.5 in batches of 10. About 2,000 posts means about 200 calls.
- Questions use Claude Sonnet 5.
- Visitors get 15 AI calls per visit and can pipeline 20 rows, so a shared link can't drain your credit. Your password removes the limits.
- Set a monthly spend limit in the Anthropic Console as a backstop.

## Tag schema

- **Photo type:** screenshot or document; ID, bill or receipt; trip or place; people or event; object or product; medical; pet.
- **Remembered:** who was in it, rough time, event, place, an object, text in the photo, a visual detail, where it came from, life context.
- **Forgotten:** exact date, album, place name, person's name, words to search with, where it came from.
- **Failure stage:** can't turn the memory into a search; search misreads the clues; too many results to judge; no way to narrow down after a miss; photo missing from library or index.
- **Also:** search tried, workarounds, outcome, stakes, verbatim quote, one-line insight.
