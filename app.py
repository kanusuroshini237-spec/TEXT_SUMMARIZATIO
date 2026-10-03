"""
Text Summarization App
Summarizes long text, .txt or .pdf files using BART models (abstractive
summarization). Long documents are split into chunks, summarized, and merged.
Runs locally. No API key needed.
"""
import streamlit as st
from pypdf import PdfReader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

st.set_page_config(page_title="Text Summarizer", page_icon="📝", layout="wide")

# ---------------- Custom CSS ----------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --primary: #6366f1;
    --primary-dark: #4f46e5;
    --accent: #8b5cf6;
    --bg: #f5f7fb;
    --card: #ffffff;
    --text: #1e293b;
    --muted: #64748b;
    --border: #e2e8f0;
    --radius: 14px;
    --shadow: 0 4px 20px rgba(15, 23, 42, 0.06);
}

/* ---------- Base ---------- */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: linear-gradient(160deg, #eef2ff 0%, var(--bg) 40%, #faf5ff 100%);
    color: var(--text);
}
.stApp p, .stApp label, .stApp span, .stApp li,
.stApp [data-testid="stMarkdownContainer"] {
    color: var(--text);
}
.block-container {
    padding-top: 2.5rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

/* ---------- Title ---------- */
h1 {
    font-weight: 700 !important;
    letter-spacing: -0.5px;
    background: linear-gradient(90deg, var(--primary), var(--accent));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    padding-bottom: 0.2rem;
}
h2, h3 {
    color: var(--text) !important;
    font-weight: 600 !important;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: var(--card);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] h2 {
    font-size: 1.15rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--primary);
    display: inline-block;
}

/* ---------- Inputs ---------- */
.stTextArea textarea {
    background: var(--card) !important;
    color: var(--text) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1rem !important;
    box-shadow: var(--shadow);
    transition: border-color .2s, box-shadow .2s;
}
.stTextArea textarea:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.18) !important;
}
div[data-baseweb="select"] > div {
    background: var(--card) !important;
    border-radius: 10px !important;
    border: 1.5px solid var(--border) !important;
    color: var(--text) !important;
}

/* ---------- Radio buttons as pills ---------- */
div[role="radiogroup"] label {
    background: var(--card);
    border: 1.5px solid var(--border);
    border-radius: 999px;
    padding: 0.3rem 1rem;
    margin-right: 0.4rem;
    transition: all .2s;
    cursor: pointer;
}
div[role="radiogroup"] label:hover {
    border-color: var(--primary);
    background: #eef2ff;
}
div[role="radiogroup"] label:has(input:checked) {
    border-color: var(--primary);
    background: #eef2ff;
    font-weight: 600;
}

/* ---------- File uploader ---------- */
[data-testid="stFileUploader"] section {
    background: var(--card);
    border: 2px dashed #c7d2fe;
    border-radius: var(--radius);
    transition: all .2s;
}
[data-testid="stFileUploader"] section:hover {
    border-color: var(--primary);
    background: #f5f7ff;
}
[data-testid="stFileUploader"] section * { color: var(--text) !important; }

/* ---------- Buttons ---------- */
.stButton > button, .stDownloadButton > button {
    border-radius: 12px;
    padding: 0.65rem 1.6rem;
    font-weight: 600;
    border: 1.5px solid var(--primary);
    background: var(--card);
    color: var(--primary);
    transition: all .2s ease;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, var(--primary), var(--accent));
    color: #fff;
    border: none;
    box-shadow: 0 6px 18px rgba(99, 102, 241, 0.35);
}
.stButton > button[kind="primary"] p { color: #fff !important; }
.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 22px rgba(99, 102, 241, 0.3);
}
.stDownloadButton > button:hover {
    background: var(--primary);
    color: #fff;
}
.stDownloadButton > button:hover p { color: #fff !important; }

/* ---------- Result cards (columns) ---------- */
[data-testid="stColumn"], [data-testid="column"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.4rem 1.6rem;
    box-shadow: var(--shadow);
}
[data-testid="stCaptionContainer"], .stApp small {
    color: var(--muted) !important;
    font-weight: 500;
}

/* ---------- Alerts ---------- */
[data-testid="stAlert"] {
    border-radius: var(--radius);
    border: none;
    box-shadow: var(--shadow);
}

/* ---------- Expander ---------- */
[data-testid="stExpander"] {
    background: var(--card);
    border: 1px solid var(--border) !important;
    border-radius: var(--radius);
    box-shadow: var(--shadow);
}

/* ---------- Progress bar ---------- */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--primary), var(--accent));
    border-radius: 999px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

MODELS = {
    "DistilBART (faster, lighter)": "sshleifer/distilbart-cnn-12-6",
    "BART Large (better quality)": "facebook/bart-large-cnn",
}

LENGTHS = {  # (min_tokens, max_tokens) per chunk
    "Short": (30, 80),
    "Medium": (60, 150),
    "Long": (100, 250),
}

CHUNK_WORDS = 600  # BART accepts ~1024 tokens; 600 words keeps us safely under


@st.cache_resource(show_spinner="Loading model (first run downloads it)...")
def load_model(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    return tokenizer, model


def chunk_text(text: str, size: int = CHUNK_WORDS):
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


def summarize_chunk(text, tokenizer, model, min_len, max_len):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    ids = model.generate(
        **inputs,
        min_length=min_len,
        max_length=max_len,
        num_beams=4,
        length_penalty=2.0,
        no_repeat_ngram_size=3,
        early_stopping=True,
    )
    return tokenizer.decode(ids[0], skip_special_tokens=True)


def summarize(text, tokenizer, model, min_len, max_len):
    chunks = chunk_text(text)
    progress = st.progress(0.0, text="Summarizing...")
    parts = []
    for i, chunk in enumerate(chunks, 1):
        # Very short chunks can't hit min_len; relax it
        words = len(chunk.split())
        parts.append(summarize_chunk(chunk, tokenizer, model, min(min_len, words // 2), max_len))
        progress.progress(i / len(chunks), text=f"Summarized chunk {i}/{len(chunks)}")
    progress.empty()

    combined = " ".join(parts)
    # If the merged summary is still long, summarize it once more
    if len(chunks) > 1 and len(combined.split()) > CHUNK_WORDS:
        combined = summarize_chunk(combined, tokenizer, model, min_len, max_len * 2)
    return combined


def read_pdf(file):
    reader = PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


# ---------------- Sidebar ----------------
st.sidebar.header("Settings")
model_name = st.sidebar.selectbox("Model", list(MODELS.keys()))
length = st.sidebar.radio("Summary length", list(LENGTHS.keys()), index=1)

# ---------------- Main ----------------
st.title("📝 Text Summarizer")
st.write("Paste text or upload a file (.txt / .pdf) to get a concise summary.")

source = st.radio("Input type", ["Paste text", "Upload file"], horizontal=True)
text = ""
if source == "Paste text":
    text = st.text_area("Your text", height=300, placeholder="Paste an article, notes, or report here...")
else:
    file = st.file_uploader("Upload .txt or .pdf", type=["txt", "pdf"])
    if file:
        text = read_pdf(file) if file.name.lower().endswith(".pdf") else file.read().decode("utf-8", errors="ignore")
        with st.expander("Preview extracted text"):
            st.write(text[:3000] + ("..." if len(text) > 3000 else ""))

if st.button("⚡ Summarize", type="primary"):
    word_count = len(text.split())
    if word_count < 50:
        st.warning("Please provide at least 50 words of text.")
    else:
        tokenizer, model = load_model(MODELS[model_name])
        min_len, max_len = LENGTHS[length]
        summary = summarize(text, tokenizer, model, min_len, max_len)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.caption(f"{word_count} words")
            st.write(text[:5000] + ("..." if len(text) > 5000 else ""))
        with col2:
            st.subheader("Summary")
            s_count = len(summary.split())
            st.caption(f"{s_count} words • {100 - s_count * 100 // word_count}% shorter")
            st.success(summary)
            st.download_button("⬇️ Download summary", summary, "summary.txt", "text/plain")