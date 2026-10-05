"""
Ask a Question Back — MVP, styled as the Ask Photos screen on an Android phone.

Path A: 20+ photos -> photos plus a narrowing question; answers become filters (max 3).
Path C: fewer than 20 -> just the photos, then "Did you find it?". "Keep looking" searches more widely.
Tap a photo to open it; "This is the one" ends the search. The user can end the search at any time.
"""

import csv
import glob
import io
import json
import os
import random
import time
import zipfile
from datetime import datetime

import streamlit as st

import mvp_logic as L
import search as S

st.set_page_config(page_title="Ask Photos prototype", page_icon="🔍", layout="centered",
                   initial_sidebar_state="collapsed")

LIB_DIR = "library"
LIB_FILE = os.path.join(LIB_DIR, "library.json")
PAGE = 21              # photos shown before "+N more" (7 rows of 3)
MAX_CALLS = 60         # AI calls per visit, protects the API budget

# ---------------------------------------------------------------- phone styling

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500&display=swap');
html, body, [class*="st-"], button, input { font-family: 'Roboto', system-ui, sans-serif !important; }
[data-testid="stAppViewContainer"] { background: #E8EAED; }
header[data-testid="stHeader"] { background: transparent; }
.block-container {
  max-width: 400px !important; background: #FFFFFF; border: 1px solid #DADCE0;
  border-radius: 32px; padding: 1.2rem 0.8rem 1.6rem !important; margin-top: 1.2rem;
}
h1 { font-size: 22px !important; font-weight: 500 !important; padding: 0 0 0.2rem !important; color: #202124; }
[data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; gap: 3px !important; }
[data-testid="stColumn"], [data-testid="column"] { min-width: 0 !important; flex: 1 1 0 !important; width: auto !important; }
[data-testid="InputInstructions"] { display: none !important; }
.st-key-grid [data-testid="stVerticalBlock"] { gap: 2px !important; }
.st-key-grid [data-testid="stHorizontalBlock"] { gap: 2px !important; }
.st-key-grid [data-testid="stColumn"], .st-key-grid [data-testid="column"] { position: relative; }
.st-key-grid img { width: 100% !important; aspect-ratio: 1 / 1; object-fit: cover; border-radius: 0; display: block; }
.st-key-grid [data-testid="stElementContainer"]:has(.stButton) { position: absolute; inset: 0; z-index: 2; margin: 0 !important; }
.st-key-grid .stButton, .st-key-grid .stButton button { width: 100% !important; height: 100% !important; }
.st-key-grid .stButton button { opacity: 0; cursor: pointer; }
.st-key-navbar { border-top: 1px solid #DADCE0; margin-top: 18px; padding-top: 8px; }
.st-key-navbar p { text-align: center; font-size: 12px; color: #5F6368; margin: 0; line-height: 1.5; }
.st-key-nav_active p { color: #174EA6; font-weight: 500; }
.st-key-nav_active { background: #D3E3FD; border-radius: 16px; padding: 2px 0; }
.st-key-searchbar [data-testid="stTextInput"] input {
  border-radius: 24px !important; background: #F1F3F4 !important; border: none !important;
  padding: 10px 16px !important; font-size: 15px !important;
}
.st-key-searchbar [data-testid="stTextInput"] > div { border: none !important; background: transparent !important; }
.st-key-searchbar [data-testid="stFormSubmitButton"] button { border-radius: 20px !important; }
.st-key-question { background: #E8F0FE; border-radius: 16px; padding: 10px 12px 6px; margin: 4px 0 6px; }
.st-key-question p { color: #174EA6; }
.st-key-confirm { background: #F1F3F4; border-radius: 16px; padding: 10px 12px; margin-top: 8px; }
.st-key-filters button { border-radius: 16px !important; }
.stButton button, [data-testid="stBaseButton-pills"], [data-testid="stBaseButton-pillsActive"] { border-radius: 18px !important; }
.small-grey { font-size: 13px; color: #5F6368; margin: 2px 2px 4px; }
.st-key-challenge { background: #E8F0FE; border-radius: 16px; padding: 12px 12px 4px; margin: 4px 0 8px; }
.st-key-challengebtn { background: #E8F0FE; border-radius: 16px; padding: 12px 12px 12px; margin-top: 8px; }
.st-key-tryblock { background: #F0F4F9; border-radius: 16px; padding: 12px 12px 6px; margin-top: 8px; }
.st-key-tryblock p { margin-bottom: 4px; }
.meta { font-size: 13px; color: #5F6368; margin: 6px 2px; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- setup

def secret(name, default=""):
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
def load_library(path, mtime):
    with open(path, encoding="utf-8") as f:
        photos = json.load(f)
    return sorted(photos, key=lambda p: p.get("date") or "", reverse=True)


if not os.path.exists(LIB_FILE):
    for z in sorted(glob.glob("library*.zip")):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(".")
        break
if not os.path.exists(LIB_FILE):
    st.error("No photo library found. Add the library folder or library.zip to the repo.")
    st.stop()

client = get_client()
LIB = load_library(LIB_FILE, os.path.getmtime(LIB_FILE))
BY_ID = {p["id"]: p for p in LIB}

ss = st.session_state
ss.setdefault("attempt", None)
ss.setdefault("log", [])
ss.setdefault("calls", 0)
ss.setdefault("target", None)
ss.setdefault("showing_target", False)


def budget_ok(cost=1):
    if client is None:
        ss["error"] = "Searching is off because no API key is set."
        return False
    if ss.calls + cost > MAX_CALLS:
        ss["error"] = "This prototype allows a limited number of searches per visit. Refresh the page to continue."
        return False
    return True


def pretty_date(d):
    try:
        return datetime.strptime(d, "%Y-%m-%d").strftime("%d %b %Y")
    except Exception:
        return d or ""


def img_path(p):
    return os.path.join(LIB_DIR, p["file"])


# ---------------------------------------------------------------- state changes

def run_search():
    query = (ss.get("q_input") or "").strip()
    if not query or not budget_ok():
        return
    try:
        res = S.search(client, LIB, query)
    except Exception as e:
        ss["error"] = f"The search didn't go through ({e}). Try again."
        return
    ss.calls += 1
    filters = L.clue_filters(res["clues"])
    photos = L.apply_filters([BY_ID[i] for i in res["ids"]], filters)
    ss.attempt = {
        "query": query, "ids": res["ids"], "filters": filters,
        "skip": [f["attr"] for f in filters], "asked": 0, "broad": False, "said_no": False,
        "answers": [], "started": time.time(), "first_count": len(photos),
        "path": "A" if L.needs_question(photos, 0) else "C",
        "shown": PAGE, "q": None, "q_key": None, "viewing": None,
        "opened": [], "backs": 0, "done": False, "outcome": None, "found_id": None,
    }


def photos_now(a):
    return L.apply_filters([BY_ID[i] for i in a["ids"]], a["filters"])


def question_now(a, photos):
    if not L.needs_question(photos, a["asked"], a["broad"]):
        return None
    key = (len(photos), a["asked"], tuple(sorted(a["skip"])))
    if a["q_key"] == key:
        return a["q"]
    q = L.pick_question(photos, set(a["skip"]))
    if q and client and ss.calls < MAX_CALLS:
        q["question"] = S.phrase(client, a["query"], q["question"])
        ss.calls += 1
    a["q"], a["q_key"] = q, key
    return q


def on_answer(widget_key):
    option = ss.get(widget_key)
    a = ss.attempt
    q = a["q"]
    if not option or not q:
        return
    f = L.answer_to_filter(q, option)
    if f:
        a["filters"].append(f)
    a["skip"].append(q["attr"])
    a["asked"] += 1
    a["answers"].append(f"{L.LABELS[q['attr']]}: {option}")
    a["shown"] = PAGE
    a["q_key"] = None


def on_remove_filter(widget_key):
    label = ss.get(widget_key)
    a = ss.attempt
    for i, f in enumerate(a["filters"]):
        if f"{L.filter_label(f)}  ✕" == label:
            a["filters"].pop(i)
            if f["attr"] in a["skip"]:
                a["skip"].remove(f["attr"])
            a["answers"].append(f"Removed: {L.filter_label(f)}")
            break
    a["shown"] = PAGE
    a["q_key"] = None


def open_photo(pid):
    a = ss.attempt
    a["viewing"] = pid
    a["opened"].append(pid)


def back_to_results():
    a = ss.attempt
    a["viewing"] = None
    a["backs"] += 1


def more():
    ss.attempt["shown"] += PAGE


def said_no():
    a = ss.attempt
    if a["broad"]:
        finish("Not found")
        return
    if not budget_ok():
        return
    try:
        res = S.search(client, LIB, a["query"], broad=True)
    except Exception as e:
        ss["error"] = f"The wider search didn't go through ({e}). Try again."
        return
    ss.calls += 1
    a["ids"] = list(dict.fromkeys(a["ids"] + res["ids"]))
    a["broad"] = True
    a["said_no"] = True
    a["answers"].append("Said: keep looking")
    a["shown"] = PAGE
    a["q_key"] = None


def finish(outcome, pid=None):
    a = ss.attempt
    a.update(done=True, outcome=outcome, found_id=pid, viewing=None)
    ss.log.append({
        "tester": ss.get("tester", ""),
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "search": a["query"],
        "first_results": a["first_count"],
        "path": a["path"],
        "said_keep_looking": "yes" if a["said_no"] else "no",
        "questions_answered": a["asked"],
        "photos_opened": len(a["opened"]),
        "back_outs": a["backs"],
        "first_photo_was_kept": ("yes" if pid and a["opened"] and a["opened"][0] == pid else
                                 "no" if a["opened"] else ""),
        "steps": " | ".join(a["answers"]),
        "outcome": outcome,
        "photo": pid or "",
        "seconds": round(time.time() - a["started"]),
        "challenge_photo": ss.target or "",
        "found_challenge_photo": ("yes" if ss.target and pid == ss.target else
                                  "no" if ss.target and pid else ""),
    })


def reset():
    ss.attempt = None
    ss["q_input"] = ""
    ss["try_pick"] = None
    ss.target = None


def start_challenge():
    ss.attempt = None
    ss["q_input"] = ""
    ss["try_pick"] = None
    ss.target = random.choice(LIB)["id"]
    ss.showing_target = True


def library_summary():
    places = [pl for pl, _ in sorted(
        ((pl, sum(1 for p in LIB if p.get("place") == pl)) for pl in {p.get("place") for p in LIB if p.get("place")}),
        key=lambda x: -x[1])]
    years = sorted({(p.get("date") or "")[:4] for p in LIB if p.get("date")})
    span = f"{years[0]}–{years[-1]}" if len(years) > 1 else (years[0] if years else "")
    place_txt = ", ".join(places[:-1]) + (f" and {places[-1]}" if len(places) > 1 else (places[0] if places else ""))
    return (f"This sample library has {len(LIB)} photos from {span}: trips and days out in {place_txt}, "
            f"with beaches, mountains, festivals and flowers.")


SUGGESTIONS = [
    "holiday photos",
    "Ganesh festival",
    "photos from Berlin",
    "flowers in the park",
    "castle in Romania",
]


def try_search():
    pick = ss.get("try_pick")
    if not pick:
        return
    ss["q_input"] = pick
    run_search()


# ---------------------------------------------------------------- sidebar (outside the phone)

with st.sidebar:
    st.header("About this prototype")
    st.write("Ask Photos with one addition: when a search returns lots of photos, it asks one question at a "
             "time to narrow them down. Every answer becomes a filter you can remove.")
    st.text_input("Tester name (for the test log)", key="tester")
    st.caption(f"Library: {len(LIB)} photos · AI calls this visit: {ss.calls} of {MAX_CALLS}")
    st.header("Test log")
    if ss.log:
        st.dataframe(ss.log, hide_index=True)
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(ss.log[0].keys()))
        w.writeheader()
        w.writerows(ss.log)
        st.download_button("Download test log (CSV)", buf.getvalue(), "test_log.csv", "text/csv")
    else:
        st.caption("Each finished search is recorded here.")


def bottom_bar():
    with st.container(key="navbar"):
        c1, c2, c3 = st.columns(3)
        c1.markdown(":material/photo_library:  \nPhotos")
        c2.markdown(":material/collections_bookmark:  \nCollections")
        with c3:
            with st.container(key="nav_active"):
                st.markdown(":material/search:  \nSearch")
    st.stop()


# ---------------------------------------------------------------- the phone screen

a = ss.attempt

# full-screen photo view
if a and not a["done"] and a["viewing"]:
    p = BY_ID[a["viewing"]]
    st.button("← Back to results", type="tertiary", on_click=back_to_results)
    st.image(img_path(p))
    st.markdown(f"<p class='meta'>{pretty_date(p.get('date'))} · {p.get('place') or 'Unknown place'}</p>",
                unsafe_allow_html=True)
    st.button("This is the one", type="primary", on_click=finish,
              args=("Found (picked a photo)", p["id"]))
    st.stop()  # full-screen photo: no bottom bar

# memory challenge: show a photo briefly, then hide it
if ss.showing_target and ss.target:
    p = BY_ID[ss.target]
    st.title("Remember this photo")
    st.markdown("<p class='small-grey'>Look closely. It disappears in 6 seconds, "
                "then you'll try to find it.</p>", unsafe_allow_html=True)
    st.image(img_path(p))
    time.sleep(6)
    ss.showing_target = False
    st.rerun()

st.title("Ask Photos")

with st.form("search", clear_on_submit=False, border=False):
    with st.container(key="searchbar"):
        c1, c2 = st.columns([5, 1.4])
        c1.text_input("Search", key="q_input", label_visibility="collapsed",
                      placeholder="Search your photos")
        c2.form_submit_button("Search", type="primary", on_click=run_search)

if ss.get("error"):
    st.error(ss.pop("error"))

if a is None:
    if ss.target:
        with st.container(key="challenge"):
            st.markdown("**Now find the photo you just saw.**")
            st.markdown("<p class='small-grey'>Describe it the way you remember it: the place, what was in it, "
                        "roughly when.</p>", unsafe_allow_html=True)
        bottom_bar()
    st.markdown(f"<p class='small-grey'>{library_summary()}</p>", unsafe_allow_html=True)
    with st.container(key="challengebtn"):
        st.markdown("**Test it the real way**")
        st.markdown("<p class='small-grey'>We show you one photo for a few seconds, then hide it. "
                    "Try to find it again from memory.</p>", unsafe_allow_html=True)
        st.button("Start the memory challenge", type="primary", on_click=start_challenge)
    with st.container(key="tryblock"):
        st.markdown("**Or just try a search:**")
        st.pills("Try searching", SUGGESTIONS, key="try_pick", label_visibility="collapsed",
                 on_change=try_search)
        st.markdown("<p class='small-grey'>Lots of results? It will ask you a question to narrow them down. "
                    "Only a few? It checks whether you found it.</p>", unsafe_allow_html=True)
    bottom_bar()

# finished
if a["done"]:
    if ss.target and a["found_id"]:
        if a["found_id"] == ss.target:
            st.success(f"You found it, in {ss.log[-1]['seconds']} seconds after {a['asked']} "
                       f"question{'s' if a['asked'] != 1 else ''}.")
            st.image(img_path(BY_ID[ss.target]))
        else:
            st.warning("Close, but not the one. This was the photo:")
            t = BY_ID[ss.target]
            st.image(img_path(t))
            st.markdown(f"<p class='meta'>{pretty_date(t.get('date'))} · {t.get('place') or 'Unknown place'}</p>",
                        unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.button("Try another", type="primary", on_click=start_challenge)
        c2.button("Free search", on_click=reset)
        bottom_bar()
    if a["found_id"]:
        p = BY_ID[a["found_id"]]
        st.success(f"Found in {ss.log[-1]['seconds']} seconds, after {a['asked']} "
                   f"question{'s' if a['asked'] != 1 else ''}.")
        st.image(img_path(p))
    elif a["outcome"].startswith("Found"):
        st.success("Glad you found it.")
    else:
        st.info("Search ended. Try describing it differently: who was there, the place, or roughly when.")
    st.button("New search", type="primary", on_click=reset)
    bottom_bar()

photos = photos_now(a)
q = question_now(a, photos)

# filters (tap to remove)
if a["filters"]:
    fkey = f"filters_{len(a['filters'])}_{a['asked']}"
    with st.container(key="filters"):
        st.pills("Filters", [f"{L.filter_label(f)}  ✕" for f in a["filters"]], key=fkey,
                 label_visibility="collapsed", on_change=on_remove_filter, args=(fkey,))

top_l, top_r = st.columns([3, 1])
top_l.markdown(f"<p class='small-grey'>{len(photos)} photo{'s' if len(photos) != 1 else ''}</p>",
               unsafe_allow_html=True)
top_r.button("End search", type="tertiary", on_click=finish, args=("Ended search",))

# the question
if q:
    qkey = f"q_{a['asked']}_{len(a['filters'])}"
    with st.container(key="question"):
        st.markdown(f"**{q['question']}**")
        opts = q["options"] + ([L.OTHER] if q["has_other"] else []) + [L.NOT_SURE]
        st.pills("Answer", opts, key=qkey, label_visibility="collapsed",
                 on_change=on_answer, args=(qkey,))

if not photos:
    st.markdown("<p class='small-grey'>No photos match. Tap a filter to remove it, or keep looking.</p>",
                unsafe_allow_html=True)

# photo grid, 3 across
shown = photos[: a["shown"]]
with st.container(key="grid"):
    for r in range(0, len(shown), 3):
        cols = st.columns(3)
        for c, p in zip(cols, shown[r:r + 3]):
            with c:
                st.image(img_path(p))
                st.button("Open", key=f"open_{p['id']}", type="tertiary",
                          on_click=open_photo, args=(p["id"],))

if len(photos) > a["shown"]:
    st.button(f"+{len(photos) - a['shown']} more", type="tertiary", on_click=more)

# few results, or questions used up: a quick check
if not q:
    with st.container(key="confirm"):
        if not photos and not a["broad"]:
            st.markdown("**Nothing matched closely.**")
            st.button("Keep looking", type="primary", on_click=said_no)
        elif not photos:
            st.markdown("**Nothing close to that in this library.**")
            st.button("End search", key="end_bottom", on_click=finish, args=("Not found",))
        else:
            st.markdown("**Is it here now?**" if a["broad"] else "**Did you find it?**")
            y, n = st.columns(2)
            y.button("Yes", type="primary", on_click=finish, args=("Found (said yes)",))
            n.button("Keep looking", on_click=said_no)

bottom_bar()
