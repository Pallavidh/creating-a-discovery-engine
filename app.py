"""Photo Recall Discovery Engine - Streamlit app."""

import io
import os

import pandas as pd
import plotly.express as px
import streamlit as st

import engine as E

st.set_page_config(page_title="Photo Recall Discovery Engine", page_icon="🔎", layout="wide")

DATA_PATH = "data/tagged.csv"
BLUE = "#1A5FB4"
PUBLIC_CALL_LIMIT = 15      # AI calls per visitor session (protects your API budget)
PUBLIC_ROW_LIMIT = 20       # rows a visitor can run through the pipeline

SUGGESTED_QUESTIONS = [
    "What kinds of old photos do users struggle to retrieve?",
    "What do people actually remember about a photo they can't find?",
    "What have they usually forgotten?",
    "How do users phrase searches when their memory is incomplete?",
    "What workarounds do people use when search fails?",
    "How do complaints about Ask Photos differ from complaints about classic search?",
]


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def secret(name: str, default: str = "") -> str:
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except Exception:
        return os.getenv(name, default)


@st.cache_resource
def get_client():
    key = secret("ANTHROPIC_API_KEY")
    if not key:
        return None
    import anthropic
    return anthropic.Anthropic(api_key=key)


@st.cache_data
def load_saved(path: str, mtime: float) -> pd.DataFrame:
    return E.load_tagged(path)


TAG_MODEL = secret("TAG_MODEL", E.TAG_MODEL_DEFAULT)
ASK_MODEL = secret("ASK_MODEL", E.ASK_MODEL_DEFAULT)
client = get_client()

ss = st.session_state
ss.setdefault("calls", 0)
ss.setdefault("admin", False)
ss.setdefault("df", None)
ss.setdefault("answer", None)


def can_call(cost: int = 1) -> bool:
    if client is None:
        st.error("The AI features are off because no API key is set. Add ANTHROPIC_API_KEY to the app secrets.")
        return False
    if ss.admin or ss.calls + cost <= PUBLIC_CALL_LIMIT:
        return True
    st.warning(f"This demo allows {PUBLIC_CALL_LIMIT} AI calls per visit. Refresh the page to start a new visit.")
    return False


def get_data():
    if ss.df is not None:
        return ss.df, "Data tagged in this session"
    if os.path.exists(DATA_PATH):
        return load_saved(DATA_PATH, os.path.getmtime(DATA_PATH)), "Saved research corpus"
    return None, None


def hbar(series: pd.Series, title: str, total: int | None = None):
    if series.empty:
        st.info(f"No data yet for: {title.lower()}")
        return
    d = series.sort_values().reset_index()
    d.columns = ["label", "count"]
    if total:
        d["text"] = d["count"].astype(str) + "  (" + (d["count"] / total * 100).round().astype(int).astype(str) + "%)"
    else:
        d["text"] = d["count"].astype(str)
    fig = px.bar(d, x="count", y="label", orientation="h", text="text")
    fig.update_traces(marker_color=BLUE, textposition="outside", cliponaxis=False)
    fig.update_layout(title=title, height=max(240, 36 * len(d) + 90),
                      margin=dict(l=10, r=70, t=50, b=10),
                      xaxis_title=None, yaxis_title=None, font=dict(size=14))
    st.plotly_chart(fig)


def show_tags(r: pd.Series):
    chips = [
        f"**Photo:** {E.PHOTO_TYPES.get(r['photo_type'], '')}",
        f"**Broke at:** {E.FAILURE_STAGES.get(r['failure_stage'], '')}",
        f"**Outcome:** {E.OUTCOMES.get(r['outcome'], '')}",
        f"**Stakes:** {E.STAKES.get(r['stakes'], '')}",
        f"**Search mode:** {E.SEARCH_MODES.get(r.get('search_mode', 'not_stated'), '')}",
    ]
    st.markdown("  \n".join(chips))
    rem = ", ".join(E.REMEMBERED_CUES[c] for c in r["remembered_cues"]) or "Not stated"
    forg = ", ".join(E.FORGOTTEN_CUES[c] for c in r["forgotten_cues"]) or "Not stated"
    work = ", ".join(E.WORKAROUNDS[c] for c in r["workarounds"]) or "Not stated"
    st.markdown(f"**Remembered:** {rem}  \n**Forgot:** {forg}  \n**Workarounds:** {work}")
    if r.get("query_tried"):
        st.markdown(f"**Searched for:** `{r['query_tried']}`")


def evidence_card(r: pd.Series):
    with st.container(border=True):
        st.markdown(f"> {r['quote'] or r['text'][:200]}")
        if r.get("insight"):
            st.markdown(f"**Insight:** {r['insight']}")
        meta = f"{r['id']} from {r.get('source', '')}"
        if r.get("url"):
            meta += f" ([open source]({r['url']}))"
        st.caption(meta)
        with st.expander("Tags and full text"):
            show_tags(r)
            st.write(r["text"])


# ---------------------------------------------------------------------------
# Header and sidebar
# ---------------------------------------------------------------------------

st.title("Photo Recall Discovery Engine")
st.write("Why do people fail to find a photo they remember but can't precisely describe? "
         "This engine reads public user posts about Google Photos, tags each one with what the "
         "person remembered, what they forgot and where retrieval broke, then compares the problems.")

df_all, data_label = get_data()

with st.sidebar:
    st.header("Data")
    if df_all is None:
        st.info("No corpus loaded yet. Use the Run the pipeline tab.")
        sources = []
    else:
        st.write(data_label)
        sources = sorted(df_all["source"].dropna().unique())
    chosen_sources = st.multiselect("Sources", sources, default=sources)

    st.header("Owner access")
    if ss.admin:
        st.success("Unlocked: no call or row limits.")
    else:
        pw = st.text_input("Password", type="password",
                           help="Lifts the demo limits so the owner can tag the full corpus.")
        if pw and pw == secret("ADMIN_PASSWORD"):
            ss.admin = True
            st.rerun()
    st.caption(f"AI calls used this visit: {ss.calls}" + ("" if ss.admin else f" of {PUBLIC_CALL_LIMIT}"))

if df_all is not None:
    df_all = df_all[df_all["source"].isin(chosen_sources)]
    rel = E.relevant_only(df_all)
else:
    rel = pd.DataFrame()

tabs = st.tabs(["Overview", "Compare problems", "Evidence", "Ask the data",
                "Run the pipeline", "How it works"])


def need_data() -> bool:
    if rel.empty:
        st.info("There is no tagged evidence to show yet. Open Run the pipeline to tag a corpus.")
        return True
    return False


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

with tabs[0]:
    if not need_data():
        n = len(rel)
        gave_up = ((rel["outcome"] == "gave_up") | rel["workarounds"].apply(lambda w: "gave_up" in w)).mean()
        c = st.columns(4)
        c[0].metric("Posts read", f"{len(df_all):,}")
        c[1].metric("About finding a photo", f"{n:,}")
        c[2].metric("Gave up", f"{gave_up:.0%}")
        c[3].metric("Urgent or practical need", f"{(rel['stakes'] == 'urgent').mean():.0%}")

        hbar(E.count_enum(rel, "failure_stage", E.FAILURE_STAGES),
             "Where retrieval breaks down", total=n)
        left, right = st.columns(2)
        with left:
            hbar(E.count_list(rel, "remembered_cues", E.REMEMBERED_CUES),
                 "What people still remember", total=n)
        with right:
            hbar(E.count_list(rel, "forgotten_cues", E.FORGOTTEN_CUES),
                 "What they have forgotten", total=n)
        left, right = st.columns(2)
        with left:
            hbar(E.count_enum(rel, "photo_type", E.PHOTO_TYPES, drop=()),
                 "Which photos people look for", total=n)
        with right:
            hbar(E.count_list(rel, "workarounds", E.WORKAROUNDS),
                 "What they do when search fails", total=n)

        modes = E.count_enum(rel, "search_mode", E.SEARCH_MODES)
        if not modes.empty:
            hbar(modes, "AI search or classic search", total=n)
            mode_stage = rel[rel["search_mode"].isin(["ask_photos", "classic"])]
            if not mode_stage.empty:
                cross = pd.crosstab(mode_stage["failure_stage"], mode_stage["search_mode"])
                cross = cross.rename(index=E.FAILURE_STAGES, columns=E.SEARCH_MODES)
                st.caption("Where each search mode breaks down")
                st.dataframe(cross)

        queries = rel[rel["query_tried"].str.len() > 0]
        if not queries.empty:
            st.subheader("Searches people report typing")
            st.dataframe(queries[["query_tried", "failure_stage", "id"]]
                         .assign(failure_stage=lambda x: x["failure_stage"].map(E.FAILURE_STAGES))
                         .rename(columns={"query_tried": "Search", "failure_stage": "Broke at", "id": "Post"}),
                         hide_index=True)


# ---------------------------------------------------------------------------
# Compare problems
# ---------------------------------------------------------------------------

with tabs[1]:
    if not need_data():
        st.subheader("Which photo types break at which stage")
        m = E.stage_type_matrix(rel)
        if m.empty:
            st.info("Not enough posts with a stated failure stage yet.")
        else:
            fig = px.imshow(m, text_auto=True, color_continuous_scale="Blues", aspect="auto")
            fig.update_layout(height=120 + 52 * len(m), xaxis_title=None, yaxis_title=None,
                              coloraxis_showscale=False, margin=dict(l=10, r=10, t=10, b=10),
                              font=dict(size=14))
            fig.update_xaxes(side="top")
            st.plotly_chart(fig)

        st.subheader("Problems ranked by opportunity")
        st.caption("Score = mentions × (1 + share who gave up) × (1 + share with an urgent need). "
                   "Frequency counts most; problems that end in giving up or block a practical task weigh up to twice as much.")
        min_n = st.slider("Minimum mentions", 1, 20, 3)
        opp = E.opportunity_table(rel, min_n=min_n)
        if opp.empty:
            st.info("No problem has that many mentions yet. Lower the minimum.")
        else:
            st.dataframe(
                opp[["problem", "mentions", "gave_up_rate", "urgent_rate", "most_remembered", "score"]],
                hide_index=True,
                column_config={
                    "problem": "Problem",
                    "mentions": "Mentions",
                    "gave_up_rate": st.column_config.NumberColumn("Gave up", format="%d%%"),
                    "urgent_rate": st.column_config.NumberColumn("Urgent", format="%d%%"),
                    "most_remembered": "Most common thing remembered",
                    "score": st.column_config.ProgressColumn(
                        "Score", min_value=0, max_value=float(opp["score"].max()), format="%.1f"),
                },
            )
            pick = st.selectbox("See the evidence behind a problem", opp["problem"])
            row = opp[opp["problem"] == pick].iloc[0]
            ev = rel[(rel["photo_type"] == row["photo_type"]) & (rel["failure_stage"] == row["failure_stage"])]
            for _, r in ev.head(8).iterrows():
                evidence_card(r)


# ---------------------------------------------------------------------------
# Evidence explorer
# ---------------------------------------------------------------------------

with tabs[2]:
    if not need_data():
        c = st.columns(3)
        f_type = c[0].multiselect("Photo type", list(E.PHOTO_TYPES), format_func=E.PHOTO_TYPES.get)
        f_stage = c[1].multiselect("Broke at", list(E.FAILURE_STAGES), format_func=E.FAILURE_STAGES.get)
        f_cue = c[2].multiselect("Remembered", list(E.REMEMBERED_CUES), format_func=E.REMEMBERED_CUES.get)
        c = st.columns(3)
        f_out = c[0].multiselect("Outcome", list(E.OUTCOMES), format_func=E.OUTCOMES.get)
        f_stakes = c[1].multiselect("Stakes", list(E.STAKES), format_func=E.STAKES.get)
        f_mode = c[2].multiselect("Search mode", list(E.SEARCH_MODES), format_func=E.SEARCH_MODES.get)
        f_text = st.text_input("Contains words")

        d = rel
        if f_type:
            d = d[d["photo_type"].isin(f_type)]
        if f_stage:
            d = d[d["failure_stage"].isin(f_stage)]
        if f_cue:
            d = d[d["remembered_cues"].apply(lambda cues: any(x in cues for x in f_cue))]
        if f_out:
            d = d[d["outcome"].isin(f_out)]
        if f_stakes:
            d = d[d["stakes"].isin(f_stakes)]
        if f_mode:
            d = d[d["search_mode"].isin(f_mode)]
        if f_text:
            d = d[d["text"].str.contains(f_text, case=False, na=False, regex=False)]

        st.write(f"**{len(d)} posts match.** Showing up to 40.")
        st.download_button("Download these posts (CSV)", E.to_csv_frame(d).to_csv(index=False),
                           "evidence.csv", "text/csv")
        for _, r in d.head(40).iterrows():
            evidence_card(r)


# ---------------------------------------------------------------------------
# Ask the data
# ---------------------------------------------------------------------------

with tabs[3]:
    if not need_data():
        st.write("Ask a research question. Answers use only the tagged posts and cite them by id.")
        cols = st.columns(len(SUGGESTED_QUESTIONS))
        for col, q in zip(cols, SUGGESTED_QUESTIONS):
            if col.button(q, key=f"sq_{q}"):
                ss.question = q
        question = st.text_area("Question", key="question", height=80)
        if st.button("Get answer", type="primary", disabled=not question.strip()) and can_call():
            with st.spinner("Reading the evidence..."):
                try:
                    ss.answer = (question, E.ask_corpus(client, rel, question, model=ASK_MODEL))
                    ss.calls += 1
                except Exception as e:
                    st.error(f"The answer failed: {e}")
        if ss.answer:
            st.markdown(f"**{ss.answer[0]}**")
            st.markdown(ss.answer[1])


# ---------------------------------------------------------------------------
# Run the pipeline
# ---------------------------------------------------------------------------

with tabs[4]:
    st.subheader("Tag a single post")
    st.write("Paste any review or comment to see how the engine reads it.")
    single = st.text_area("Post text", height=120,
                          placeholder="e.g. Trying to find a photo of a prescription from last winter. "
                                      "Searched 'medicine' and 'tablet', got hundreds of pill photos...")
    if st.button("Tag this post", disabled=not single.strip()) and can_call():
        with st.spinner("Tagging..."):
            try:
                out = E.tag_batch(client, [{"id": "demo", "source": "pasted", "text": single}], model=TAG_MODEL)
                ss.calls += 1
                tag = out.get("demo")
                if not tag:
                    st.error("The model returned no tags. Try again.")
                elif not tag["relevant"]:
                    st.info("Not about finding a specific photo, so the engine filters it out.")
                else:
                    st.success(tag["insight"] or "Tagged.")
                    show_tags(pd.Series(tag))
            except Exception as e:
                st.error(f"Tagging failed: {e}")

    st.divider()
    st.subheader("Run the full pipeline on a CSV")
    st.write("Upload the corpus from the collection notebook. Required columns: `id`, `text`. "
             "Optional: `source`, `url`, `date`, `rating`.")
    if not ss.admin:
        st.caption(f"Visitors can run the first {PUBLIC_ROW_LIMIT} rows. The owner can run everything.")
    up = st.file_uploader("Corpus CSV", type="csv")
    if up is not None:
        corpus = pd.read_csv(up, dtype={"id": str})
        missing = {"id", "text"} - set(corpus.columns)
        if missing:
            st.error(f"The file is missing these columns: {', '.join(sorted(missing))}")
        else:
            if not ss.admin:
                corpus = corpus.head(PUBLIC_ROW_LIMIT)
            n_batches = -(-len(corpus) // 10)
            st.write(f"{len(corpus):,} rows ready, {n_batches} AI calls.")
            if st.button("Run pipeline", type="primary") and can_call(n_batches):
                bar = st.progress(0.0, text="Starting...")
                tagged = E.tag_corpus(
                    client, corpus, model=TAG_MODEL,
                    progress_cb=lambda d, t, f: bar.progress(d / t, text=f"Batch {d} of {t}, {f} failed"),
                )
                ss.calls += n_batches
                tagged = E.load_tagged(io.StringIO(E.to_csv_frame(tagged).to_csv(index=False)))
                ss.df = tagged
                untagged = int((~tagged["tagged"].astype(str).str.lower().eq("true")).sum())
                st.success(f"Tagged {len(tagged) - untagged:,} posts. "
                           f"{int(tagged['relevant'].sum()):,} are about finding a photo. "
                           "The other tabs now show this data.")
                if untagged:
                    st.warning(f"{untagged} posts could not be tagged. Run the file again to retry them.")
    if ss.df is not None:
        st.download_button("Download tagged CSV", E.to_csv_frame(ss.df).to_csv(index=False),
                           "tagged.csv", "text/csv", type="primary")
        st.caption("To make this the saved corpus for all visitors, commit the file as data/tagged.csv.")


# ---------------------------------------------------------------------------
# How it works
# ---------------------------------------------------------------------------

with tabs[5]:
    st.subheader("Pipeline")
    st.markdown(
        """
1. **Collect.** A notebook pulls Play Store and App Store reviews, Reddit posts and comments, and any hand-collected forum or YouTube threads into one CSV. App reviews are pre-filtered for finding-related words.
2. **Filter.** Claude keeps only posts about trying to find a specific existing photo. Storage, pricing and backup complaints are dropped.
3. **Extract.** For each kept post, Claude records the photo type, what the person remembered, what they forgot, the search they tried, where retrieval broke, their workaround, the outcome, the stakes, whether they were using AI or classic search, and a verbatim quote.
4. **Compare.** Posts are aggregated into a photo type × failure stage grid and ranked by an opportunity score.
5. **Ask.** Research questions are answered from the tagged evidence only, with every claim cited to a post.
"""
    )
    st.subheader("Where retrieval breaks: the five stages")
    st.markdown(
        """
- **Can't turn the memory into a search:** the person remembers something, but not in words search understands.
- **Search misreads the clues:** they give clues, and results are wrong or empty.
- **Too many results to judge:** plausible results come back, but they can't tell which is the one.
- **No way to narrow down after a miss:** a failed search is a dead end, with nothing to build on.
- **Photo missing from library or index:** it was never backed up, was deleted, or sits in another app.
"""
    )
    st.subheader("Models")
    st.write(f"Tagging: `{TAG_MODEL}`. Answering questions: `{ASK_MODEL}`.")
    st.caption("All sources are public posts. The engine stores short quotes and links, not user names.")
