"""
Photo Recall Discovery Engine - core logic.

Turns raw user posts (app reviews, Reddit, forums) into structured evidence
about WHY people fail to retrieve photos they only partly remember.

Used by app.py, and runnable on its own for bulk tagging:
    ANTHROPIC_API_KEY=... python engine.py corpus.csv data/tagged.csv
"""

from __future__ import annotations

import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

TAG_MODEL_DEFAULT = "claude-haiku-4-5-20251001"   # cheap + fast for bulk tagging
ASK_MODEL_DEFAULT = "claude-sonnet-5"             # better reasoning for synthesis

MAX_POST_CHARS = 1500

# ---------------------------------------------------------------------------
# Tagging schema (code -> human label). The codes are what the model returns.
# ---------------------------------------------------------------------------

PHOTO_TYPES = {
    "screenshot_or_document": "Screenshot or document",
    "id_bill_or_receipt": "ID, bill or receipt",
    "trip_or_place": "Trip or place",
    "people_or_event": "People or event",
    "object_or_product": "Object or product",
    "medical": "Medical",
    "pet": "Pet",
    "other_or_unclear": "Other or unclear",
}

REMEMBERED_CUES = {
    "person": "Who was in it",
    "rough_time": "Rough time period",
    "event_or_occasion": "Event or occasion",
    "place": "Place or trip",
    "object": "An object in it",
    "text_in_photo": "Text in the photo",
    "visual_detail": "Colour or visual detail",
    "source_app": "Where it came from (e.g. WhatsApp)",
    "life_context": "What was happening in life then",
}

FORGOTTEN_CUES = {
    "exact_date": "Exact date",
    "album_or_folder": "Album or folder",
    "place_name": "Place name",
    "person_name": "Person's name",
    "exact_words": "Words to search with",
    "source_app": "Where it came from",
}

FAILURE_STAGES = {
    "cant_express": "Can't turn the memory into a search",
    "misinterpreted": "Search misreads the clues",
    "cant_evaluate": "Too many results to judge",
    "cant_refine": "No way to narrow down after a miss",
    "not_in_library": "Photo missing from library or index",
    "not_stated": "Not stated",
}

WORKAROUNDS = {
    "scroll_timeline": "Scrolled the timeline",
    "other_app": "Searched another app (WhatsApp, Drive, Gmail)",
    "asked_someone": "Asked someone who was there",
    "date_or_map": "Used date, map or places view",
    "rephrased": "Tried many search phrasings",
    "switched_search_mode": "Switched Ask Photos / classic search",
    "gave_up": "Gave up",
}

OUTCOMES = {
    "found": "Found it",
    "gave_up": "Gave up",
    "unknown": "Unknown or still looking",
}

SEARCH_MODES = {
    "ask_photos": "Ask Photos (AI search)",
    "classic": "Classic keyword search",
    "both": "Compares both modes",
    "not_stated": "Not stated",
}

STAKES = {
    "urgent": "Urgent or practical need",
    "sentimental": "Sentimental",
    "casual": "Casual",
}

LIST_FIELDS = ["remembered_cues", "forgotten_cues", "workarounds"]
ENUM_FIELDS = {
    "photo_type": (PHOTO_TYPES, "other_or_unclear"),
    "failure_stage": (FAILURE_STAGES, "not_stated"),
    "outcome": (OUTCOMES, "unknown"),
    "stakes": (STAKES, "casual"),
    "search_mode": (SEARCH_MODES, "not_stated"),
}
LIST_ENUMS = {
    "remembered_cues": REMEMBERED_CUES,
    "forgotten_cues": FORGOTTEN_CUES,
    "workarounds": WORKAROUNDS,
}
TAG_COLUMNS = ["relevant", "photo_type", "remembered_cues", "forgotten_cues",
               "query_tried", "failure_stage", "workarounds", "outcome",
               "stakes", "search_mode", "quote", "insight"]

SYSTEM_PROMPT = """You are a UX research analyst studying one question: why do Google Photos users fail to find a specific photo they only partly remember?

You will receive user posts (app reviews, Reddit posts and comments, forum threads), each with an id. Call record_tags exactly once, with one entry for EVERY post id.

Rules
- relevant = true only if the text describes trying to find, search for or retrieve a specific existing photo or video, or describes how search/finding works for them. Storage limits, pricing, backup speed, editing, sharing or generic praise with no finding task => relevant = false. For irrelevant posts use: photo_type "other_or_unclear", empty lists, failure_stage "not_stated", outcome "unknown", stakes "casual", search_mode "not_stated", empty strings.
- Tag only what the text says. Never guess. Unstated => empty list or "not_stated".
- remembered_cues: what the person could still recall about the target photo.
- forgotten_cues: what they say they could not recall or did not know.
- failure_stage: the main point where retrieval broke down:
  cant_express   - could not turn the memory into words or a query the app handles
  misinterpreted - gave clues, but search returned wrong results or nothing
  cant_evaluate  - results came back, but too many or too similar to spot the right one
  cant_refine    - after a failed search, no way to narrow down or build on it
  not_in_library - never backed up, deleted, stuck in another app, or not indexed
- workarounds: what they did instead, if stated.
- search_mode: which search the post is about. ask_photos = the AI / Ask Photos / Gemini experience; classic = the old keyword search; both = the post compares the two or describes switching between them; not_stated = the post does not say.
- stakes: urgent = needed it for a practical task (ID, bill, prescription, proof, work); sentimental = memories, people who passed away, milestones; casual = otherwise.
- query_tried: search words they report typing, copied exactly; else "".
- quote: the single most telling fragment, copied exactly, max 25 words.
- insight: one plain sentence, max 20 words, on what this post reveals about retrieval failure; "" if irrelevant.
Posts may be in English, Hindi or Hinglish; write insight in English.
- failure_stage: assign a stage ONLY when the text describes what actually happened in a specific attempt: what they looked for, what they typed, what came back, or what they did next. A general complaint that search is bad, broken or AI-ruined, with no attempt described, gets "not_stated". Never infer "misinterpreted" from a rant.
- Problems with editing, loading, downloading, albums or the UI are not retrieval failures. If the post is about one of those, relevant = false."""


def _tool_schema() -> dict:
    item = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "relevant": {"type": "boolean"},
            "photo_type": {"type": "string", "enum": list(PHOTO_TYPES)},
            "remembered_cues": {"type": "array", "items": {"type": "string", "enum": list(REMEMBERED_CUES)}},
            "forgotten_cues": {"type": "array", "items": {"type": "string", "enum": list(FORGOTTEN_CUES)}},
            "query_tried": {"type": "string"},
            "failure_stage": {"type": "string", "enum": list(FAILURE_STAGES)},
            "workarounds": {"type": "array", "items": {"type": "string", "enum": list(WORKAROUNDS)}},
            "outcome": {"type": "string", "enum": list(OUTCOMES)},
            "stakes": {"type": "string", "enum": list(STAKES)},
            "search_mode": {"type": "string", "enum": list(SEARCH_MODES)},
            "quote": {"type": "string"},
            "insight": {"type": "string"},
        },
        "required": ["id", "relevant", "photo_type", "remembered_cues", "forgotten_cues",
                     "query_tried", "failure_stage", "workarounds", "outcome", "stakes",
                     "search_mode", "quote", "insight"],
    }
    return {
        "name": "record_tags",
        "description": "Record structured research tags for every post in the batch.",
        "input_schema": {
            "type": "object",
            "properties": {"posts": {"type": "array", "items": item}},
            "required": ["posts"],
        },
    }


TOOL = _tool_schema()


# ---------------------------------------------------------------------------
# Tagging
# ---------------------------------------------------------------------------

def _clean(text) -> str:
    text = "" if pd.isna(text) else str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_POST_CHARS]


def _normalise(tag: dict) -> dict:
    """Coerce model output into the schema so bad values never break charts."""
    out = {"relevant": bool(tag.get("relevant", False))}
    for field, (allowed, default) in ENUM_FIELDS.items():
        val = tag.get(field, default)
        out[field] = val if val in allowed else default
    for field, allowed in LIST_ENUMS.items():
        vals = tag.get(field) or []
        if isinstance(vals, str):
            vals = [vals]
        out[field] = sorted({v for v in vals if v in allowed})
    for field in ("query_tried", "quote", "insight"):
        out[field] = str(tag.get(field, "") or "").strip()
    return out


def tag_batch(client, rows: list[dict], model: str = TAG_MODEL_DEFAULT, retries: int = 3) -> dict:
    """rows: [{id, source, text}] -> {id: normalised tag dict}"""
    body = "\n\n".join(f"<post id=\"{r['id']}\" source=\"{r.get('source', '')}\">\n{_clean(r['text'])}\n</post>"
                       for r in rows)
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=4000,
                system=SYSTEM_PROMPT,
                tools=[TOOL],
                tool_choice={"type": "tool", "name": "record_tags"},
                messages=[{"role": "user", "content": body}],
            )
            for block in resp.content:
                if getattr(block, "type", "") == "tool_use":
                    posts = block.input.get("posts", [])
                    return {str(p.get("id")): _normalise(p) for p in posts if p.get("id") is not None}
            raise ValueError("No tool output returned")
        except Exception as e:  # network, rate limit, bad output
            last_err = e
            time.sleep(2 ** attempt * 2)
    raise RuntimeError(f"Batch failed after {retries} attempts: {last_err}")


def tag_corpus(client, df: pd.DataFrame, model: str = TAG_MODEL_DEFAULT,
               batch_size: int = 10, workers: int = 4, progress_cb=None) -> pd.DataFrame:
    """Tag every row of a corpus. df needs columns: id, text (source/url/date/rating optional)."""
    df = df.copy()
    df["id"] = df["id"].astype(str)
    df = df[df["text"].fillna("").str.strip().str.len() > 0].drop_duplicates("id")
    rows = df[["id", "text"] + (["source"] if "source" in df else [])].to_dict("records")
    batches = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]

    results, failed = {}, 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(tag_batch, client, b, model) for b in batches]
        for done, fut in enumerate(as_completed(futures), start=1):
            try:
                results.update(fut.result())
            except Exception:
                failed += 1
            if progress_cb:
                progress_cb(done, len(batches), failed)

    tags = pd.DataFrame([{"id": k, **v} for k, v in results.items()])
    if tags.empty:
        tags = pd.DataFrame(columns=["id"] + TAG_COLUMNS)
    merged = df.merge(tags, on="id", how="left")
    merged["tagged"] = merged["relevant"].notna()
    return merged


# ---------------------------------------------------------------------------
# Storage helpers (lists are stored as "a|b|c" in CSV)
# ---------------------------------------------------------------------------

def to_csv_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for f in LIST_FIELDS:
        if f in out:
            out[f] = out[f].apply(lambda v: "|".join(v) if isinstance(v, (list, tuple)) else ("" if pd.isna(v) else str(v)))
    return out


def load_tagged(src) -> pd.DataFrame:
    df = pd.read_csv(src, dtype={"id": str})
    for f in LIST_FIELDS:
        if f in df:
            df[f] = df[f].fillna("").apply(lambda s: [x for x in str(s).split("|") if x])
    if "relevant" in df:
        df["relevant"] = df["relevant"].astype(str).str.lower().isin(["true", "1"])
    for f in ("quote", "insight", "query_tried", "url", "source", "text"):
        if f in df:
            df[f] = df[f].fillna("")
    if "source" not in df:
        df["source"] = "unknown"
    return df


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def relevant_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["relevant"] == True].copy()  # noqa: E712


def count_enum(df: pd.DataFrame, field: str, labels: dict, drop=("not_stated",)) -> pd.Series:
    s = df[field].value_counts()
    s = s[[k for k in s.index if k not in drop]]
    return s.rename(index=labels)


def count_list(df: pd.DataFrame, field: str, labels: dict) -> pd.Series:
    s = df[field].explode().dropna().value_counts()
    return s.rename(index=labels)


def stage_type_matrix(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["failure_stage"] != "not_stated"]
    m = pd.crosstab(d["photo_type"], d["failure_stage"])
    m = m.reindex(index=[k for k in PHOTO_TYPES if k in m.index],
                  columns=[k for k in FAILURE_STAGES if k in m.columns])
    return m.rename(index=PHOTO_TYPES, columns=FAILURE_STAGES)


def opportunity_table(df: pd.DataFrame, min_n: int = 3) -> pd.DataFrame:
    """Score each (photo type x failure stage) problem.

    score = mentions x (1 + gave-up rate) x (1 + urgent rate)
    Frequency matters most; problems that make people give up, or block an
    urgent task, get up to 2x weight each.
    """
    d = df[df["failure_stage"] != "not_stated"].copy()
    if d.empty:
        return pd.DataFrame()
    d["gave_up"] = (d["outcome"] == "gave_up") | d["workarounds"].apply(lambda w: "gave_up" in w)
    d["is_urgent"] = d["stakes"] == "urgent"
    g = d.groupby(["photo_type", "failure_stage"]).agg(
        mentions=("id", "count"),
        gave_up_rate=("gave_up", "mean"),
        urgent_rate=("is_urgent", "mean"),
    ).reset_index()
    g = g[g["mentions"] >= min_n]
    g["score"] = g["mentions"] * (1 + g["gave_up_rate"]) * (1 + g["urgent_rate"])
    top_cue = (d.explode("remembered_cues").dropna(subset=["remembered_cues"])
               .groupby(["photo_type", "failure_stage"])["remembered_cues"]
               .agg(lambda s: REMEMBERED_CUES.get(s.value_counts().index[0], "")))
    g = g.merge(top_cue.rename("most_remembered").reset_index(), how="left",
                on=["photo_type", "failure_stage"])
    g["most_remembered"] = g["most_remembered"].fillna("")
    g["problem"] = g["photo_type"].map(PHOTO_TYPES) + " / " + g["failure_stage"].map(FAILURE_STAGES)
    g = g.sort_values("score", ascending=False).reset_index(drop=True)
    g["gave_up_rate"] = (g["gave_up_rate"] * 100).round().astype(int)
    g["urgent_rate"] = (g["urgent_rate"] * 100).round().astype(int)
    g["score"] = g["score"].round(1)
    return g


def _evidence_line(r) -> str:
    parts = [
        f"[{r['id']}] {r.get('source', '')}",
        f"type={r['photo_type']}",
        f"stage={r['failure_stage']}",
        f"remembered={','.join(r['remembered_cues']) or '-'}",
        f"forgot={','.join(r['forgotten_cues']) or '-'}",
        f"workarounds={','.join(r['workarounds']) or '-'}",
        f"outcome={r['outcome']}",
        f"stakes={r['stakes']}",
        f"search_mode={r['search_mode']}",
    ]
    if r.get("query_tried"):
        parts.append(f"query=\"{r['query_tried']}\"")
    parts.append(f"quote=\"{r['quote']}\"")
    if r.get("insight"):
        parts.append(f"insight={r['insight']}")
    return " | ".join(parts)


ASK_SYSTEM = """You are a product research analyst. Answer the question using ONLY the tagged evidence provided, which comes from real public user posts about finding photos in Google Photos.
- Lead with the direct answer in 1-2 sentences.
- Back it with counts from the evidence where you can (e.g. "18 of 140 posts").
- Cite post ids in square brackets, e.g. [rd-0042], for every claim; quote short fragments where useful.
- Point out patterns across posts and any contradictions.
- If the evidence is thin (under ~5 posts), say so plainly. Never invent posts or numbers.
 - Never return an empty answer. Every question gets a response.
- If the question asks for a ranking, a total, a count across the whole corpus, or "which is most common", say the Overview tab already shows these as charts — what people remembered, what they forgot, where retrieval broke, which photos they looked for, and what they did when search failed — and point the reader there. Then answer as far as the cited evidence allows.
- If you cannot answer as asked for any other reason, say why in one line, then suggest two questions this evidence can answer, each on its own line starting with "Try asking:".
- Never repeat or restate the question. Open with the answer itself.
- Cite at most 5 post ids per claim. Give a count rather than listing every matching id.
- Keep it under 250 words. Use short paragraphs or a short list."""



def ask_corpus(client, df: pd.DataFrame, question: str, model: str = ASK_MODEL_DEFAULT,
               max_rows: int = 600) -> str:
    d = relevant_only(df)
    if d.empty:
        return "There is no tagged evidence yet. Run the pipeline first."
    if len(d) > max_rows:  # simple keyword ranking to keep context focused
        words = {w for w in re.findall(r"[a-z]{4,}", question.lower())}
        blob = (d["quote"] + " " + d["insight"] + " " + d["text"]).str.lower()
        d = d.assign(_hits=blob.apply(lambda t: sum(w in t for w in words)))
        d = d.sort_values("_hits", ascending=False).head(max_rows)
    evidence = "\n".join(_evidence_line(r) for _, r in d.iterrows())
    resp = client.messages.create(
        model=model,
        max_tokens=3000,
        system=ASK_SYSTEM,
        messages=[{"role": "user", "content": f"<evidence count=\"{len(d)}\">\n{evidence}\n</evidence>\n\nQuestion: {question}"}],
    )
    return "".join(getattr(b, "text", "") for b in resp.content).strip()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python engine.py corpus.csv data/tagged.csv")
        sys.exit(1)
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    corpus = pd.read_csv(sys.argv[1], dtype={"id": str})
    tagged = tag_corpus(
        client, corpus,
        progress_cb=lambda d, t, f: print(f"\rBatch {d}/{t} (failed: {f})", end="", flush=True),
    )
    to_csv_frame(tagged).to_csv(sys.argv[2], index=False)
    rel = int(tagged["relevant"].fillna(False).astype(bool).sum())
    print(f"\nSaved {len(tagged)} rows ({rel} about photo retrieval) to {sys.argv[2]}")
