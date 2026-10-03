# 📝 Text Summarizer

A local, offline-friendly text summarization app built with **Streamlit** and **Hugging Face Transformers**. Paste text or upload a `.txt` / `.pdf` file and get a concise abstractive summary using BART models. No API key required.

## Features

- **Abstractive summarization** with BART (`facebook/bart-large-cnn`) or the lighter DistilBART (`sshleifer/distilbart-cnn-12-6`)
- **Paste text or upload files** (`.txt` and `.pdf`)
- **Handles long documents** by splitting into ~600-word chunks, summarizing each, and merging (with an extra pass if the merged result is still long)
- **Adjustable summary length:** Short, Medium, or Long
- **Side-by-side view** of the original text and the summary, with word count and percent reduction
- **Download** the summary as `summary.txt`
- **Custom styled UI** (gradient theme, card layout, pill-style radios)
- Runs **entirely on your machine**

## Requirements

- Python 3.9+
- ~1 GB free disk space for the BART Large model (DistilBART is smaller)

## Installation

```bash
# 1. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install streamlit transformers torch pypdf
```

Or save this as `requirements.txt`:

```
streamlit
transformers
torch
pypdf
```

and run `pip install -r requirements.txt`.

## Usage

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

1. Choose a **model** and **summary length** in the sidebar.
2. Select **Paste text** or **Upload file**.
3. Provide at least **50 words** of text.
4. Click **⚡ Summarize**.
5. Optionally click **⬇️ Download summary**.

> The first run downloads the selected model from Hugging Face, so it may take a few minutes. Later runs load it from the local cache.

## Settings

| Model | Speed | Quality |
|-------|-------|---------|
| DistilBART | Faster, lighter | Good |
| BART Large | Slower | Better |

| Length | Tokens per chunk (min–max) |
|--------|----------------------------|
| Short  | 30–80 |
| Medium | 60–150 |
| Long   | 100–250 |

Length applies **per chunk**, so longer documents produce longer overall summaries.

## How It Works

1. Text is extracted (PDFs via `pypdf`).
2. It is split into chunks of 600 words, which keeps each chunk under BART's ~1024-token limit.
3. Each chunk is summarized with beam search (`num_beams=4`, `no_repeat_ngram_size=3`).
4. Chunk summaries are joined. If the result exceeds 600 words, it is summarized once more.

## Customization

- **Models:** edit the `MODELS` dictionary in `app.py`.
- **Length presets:** edit the `LENGTHS` dictionary.
- **Chunk size:** change `CHUNK_WORDS`.
- **Styling:** edit the `CUSTOM_CSS` block near the top of `app.py` (colors are CSS variables under `:root`).

## Limitations

- **Scanned PDFs** (images of text) won't work, since there is no OCR. Only PDFs with selectable text are supported.
- BART is trained mainly on **English news**, so it works best on English articles and reports.
- Summaries are abstractive and may occasionally contain inaccuracies, so verify important details against the source.
- CPU inference on very long documents with BART Large can be slow. Use DistilBART for speed, or a GPU if available.

## Project Structure

```
.
├── app.py          # Streamlit application
└── README.md
```

## Tech Stack

[Streamlit](https://streamlit.io) · [Transformers](https://huggingface.co/docs/transformers) · [PyTorch](https://pytorch.org) · [pypdf](https://pypdf.readthedocs.io)
