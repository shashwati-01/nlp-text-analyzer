import io
from html import escape

import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
import pandas as pd
import spacy
import streamlit as st

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer


st.set_page_config(page_title="NLP Text Analyzer", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")


@st.cache_resource(show_spinner=False)
def load_nlp_resources():
    """Downloads required data once and reuses the loaded spaCy pipeline."""
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
    return spacy.load("en_core_web_sm")


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink:#18213b; --muted:#66708a; --line:rgba(92,105,150,.16); }
    .stApp { background:radial-gradient(circle at 6% 5%,rgba(105,161,255,.18),transparent 23rem),radial-gradient(circle at 90% 11%,rgba(230,117,194,.14),transparent 20rem),radial-gradient(circle at 55% 92%,rgba(255,180,97,.13),transparent 25rem),#fbfcff; color:var(--ink); font-family:'DM Sans',sans-serif; }
    #MainMenu, footer { visibility:hidden; }
    [data-testid="stHeader"] { visibility:visible; background:transparent; }
    [data-testid="stSidebarCollapsedControl"] { display:flex; color:#4c74e8; }
    .block-container { padding-top:2.4rem; padding-bottom:2rem; max-width:1300px; }
    .particle { position:fixed; border-radius:50%; pointer-events:none; filter:blur(1px); opacity:.55; z-index:0; animation:float 9s ease-in-out infinite; }
    .p1 { width:12px; height:12px; background:#869aff; top:19%; left:6%; }.p2 { width:9px; height:9px; background:#f39fc9; top:61%; right:7%; animation-delay:-3s; }.p3 { width:7px; height:7px; background:#f6b66b; top:31%; right:15%; animation-delay:-5s; }
    @keyframes float { 50% { transform:translateY(-22px) translateX(9px); opacity:.9; } }
    .hero { position:relative; overflow:hidden; padding:3.4rem 3.5rem; border:1px solid var(--line); border-radius:28px; background:linear-gradient(115deg,rgba(255,255,255,.9),rgba(247,248,255,.65)); box-shadow:0 20px 60px rgba(74,87,142,.10); margin-bottom:1.8rem; }
    .hero:after { content:''; position:absolute; width:280px; height:280px; border-radius:50%; background:linear-gradient(135deg,rgba(89,131,255,.25),rgba(218,112,204,.14),rgba(255,170,78,.18)); right:-90px; top:-120px; filter:blur(8px); }
    .eyebrow { color:#5b69ac; font-weight:700; letter-spacing:.13em; font-size:.76rem; text-transform:uppercase; }
    .project-info { display:flex; flex-wrap:wrap; gap:.55rem .75rem; margin:1rem 0 .85rem; }
    .project-info span { color:#4f5d83; font-size:.92rem; font-weight:600; background:rgba(255,255,255,.70); border:1px solid rgba(92,105,150,.14); border-radius:10px; padding:.48rem .72rem; }
    .hero h1 { font-family:'Space Grotesk',sans-serif; font-size:clamp(2.2rem,5vw,4.25rem); letter-spacing:-.055em; margin:.6rem 0 .7rem; line-height:1.02; }
    .hero p { max-width:720px; color:var(--muted); font-size:1.04rem; line-height:1.7; margin:0; }
    .section-title { font-family:'Space Grotesk',sans-serif; font-size:1.35rem; margin:.3rem 0 .75rem; }
    .glass-card { background:rgba(255,255,255,.72); border:1px solid var(--line); backdrop-filter:blur(14px); border-radius:20px; padding:1.35rem 1.45rem; margin:.8rem 0 1.2rem; box-shadow:0 10px 30px rgba(72,84,130,.06); }
    .glass-card h3 { font-family:'Space Grotesk',sans-serif; font-size:1.08rem; margin:0 0 .65rem; }
    [data-testid='stMetric'] { background:rgba(255,255,255,.58); border:1px solid var(--line); border-radius:18px; padding:1rem; }
    [data-testid='stMetricLabel'] { color:var(--muted); }
    .stButton>button { border:0; border-radius:13px; color:white; font-family:'DM Sans',sans-serif; font-weight:700; padding:.72rem 1.3rem; background:linear-gradient(100deg,#4c74e8,#9862dc 48%,#e075b5 72%,#ec985e); box-shadow:0 10px 22px rgba(108,92,206,.24); transition:transform .2s ease,box-shadow .2s ease; }
    .stButton>button:hover { color:white; transform:translateY(-2px); box-shadow:0 14px 30px rgba(108,92,206,.30); }
    .stTextArea textarea { border:1px solid var(--line); border-radius:18px; background:rgba(255,255,255,.72); font-size:1rem; }
    [data-testid='stSidebar'] { background:rgba(247,249,255,.92); border-right:1px solid var(--line); }
    [data-testid='stSidebar'] h2, .sidebar-brand { font-family:'Space Grotesk',sans-serif; }
    .sidebar-brand { font-size:1.45rem; font-weight:700; margin:.8rem 0 1.8rem; }.tech { display:inline-block; margin:.18rem .15rem .18rem 0; padding:.33rem .62rem; border-radius:999px; background:#eef1ff; color:#4c5ca5; font-size:.79rem; font-weight:700; }
    .footer { text-align:center; color:var(--muted); margin-top:2rem; font-size:.9rem; }.token-output { color:#39435f; line-height:1.75; word-break:break-word; }
    @media(max-width:700px) { .hero { padding:2.2rem 1.55rem; } .block-container { padding-top:1.2rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)

# Keep particle markup in its own HTML-enabled render call so it is never displayed as text.
st.markdown("<div class='particle p1'></div><div class='particle p2'></div><div class='particle p3'></div>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<div class='sidebar-brand'>🧠 Neural Lens</div>", unsafe_allow_html=True)
    st.header("Project Information")
    st.markdown("**Developer:**     Shashwati")
    st.divider()
    st.header("Technology")
    st.markdown("<span class='tech'>Python</span><span class='tech'>NLTK</span><span class='tech'>spaCy</span><span class='tech'>Streamlit</span>", unsafe_allow_html=True)
    st.caption("A refined workspace for practical natural language processing.")

st.markdown("""<section class='hero'>
<div class='eyebrow'>Intelligent language workspace</div>
<div class='project-info'>
    <span>👩‍💻 Developer: Shashwati Joshi</span>
    <span>📅 Date: 22/07/2026</span>
    <span>🔢 Roll No: 46</span>
</div>
<h1>🧠 NLP Text Analyzer</h1>
<p><strong>Natural Language Processing using NLTK and spaCy</strong><br>A beautiful modern interface for performing sentence segmentation, tokenization, stop word removal, stemming, lemmatization, POS tagging, Named Entity Recognition, Dependency Parsing, and Noun Phrase Chunking.</p>
</section>""", unsafe_allow_html=True)

if "text_input" not in st.session_state:
    st.session_state.text_input = ""

st.markdown("<div class='section-title'>Your text</div>", unsafe_allow_html=True)
text = st.text_area("Enter text for analysis", placeholder="Enter any paragraph here...", height=210, key="text_input", label_visibility="collapsed")
button_col, reset_col, _ = st.columns([1.25, 1, 5])
with button_col:
    analyze = st.button("✦ Analyze Text", use_container_width=True)

def reset_text():
    st.session_state.text_input = ""

with reset_col:
    st.button("Reset", use_container_width=True, on_click=reset_text)


def card(title, content):
    st.markdown(f"<div class='glass-card'><h3>{title}</h3>{content}</div>", unsafe_allow_html=True)


if analyze:
    if not text.strip():
        st.warning("Please enter a paragraph before analyzing.")
    else:
        with st.spinner("Analyzing your text with NLTK and spaCy..."):
            try:
                nlp = load_nlp_resources()
            except OSError:
                st.error("The spaCy model `en_core_web_sm` is not installed. Run `python -m spacy download en_core_web_sm` and restart the app.")
                st.stop()

            # Step 1: Sentence Segmentation
            sentences = sent_tokenize(text)
            # Step 2: Word Tokenization
            words = word_tokenize(text)
            # Step 3: Stop Word Removal
            stop_words = set(stopwords.words("english"))
            filtered_words = [word for word in words if word.lower() not in stop_words]
            # Step 4: Stemming
            stemmer = PorterStemmer()
            stemmed_words = [stemmer.stem(word) for word in filtered_words]
            # Step 5: Lemmatization
            lemmatizer = WordNetLemmatizer()
            lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
            # Step 6: POS Tagging
            pos_tags = nltk.pos_tag(words)
            # Step 7: Named Entity Recognition (NER)
            doc = nlp(text)
            # Step 8: Dependency Parsing
            dependencies = [(token.text, token.dep_, token.head.text) for token in doc]
            # Step 9: Chunking (Noun Phrases)
            noun_phrases = [chunk.text for chunk in doc.noun_chunks]

        st.success("Analysis complete — your language insights are ready.")
        metric_1, metric_2, metric_3, metric_4 = st.columns(4)
        metric_1.metric("Words", len(words)); metric_2.metric("Sentences", len(sentences)); metric_3.metric("Entities", len(doc.ents)); metric_4.metric("Noun Phrases", len(noun_phrases))

        card("📄 Original Text", f"<div class='token-output'>{escape(text)}</div>")
        card("✂ Sentence Segmentation", "".join(f"<div class='token-output'><b>{i}.</b> {escape(sent)}</div>" for i, sent in enumerate(sentences, 1)))
        card("🔤 Word Tokenization", f"<div class='token-output'>{escape(', '.join(words))}</div>")
        card("🚫 Stop Word Removal", f"<div class='token-output'>{escape(', '.join(filtered_words))}</div>")
        card("🌱 Stemming", f"<div class='token-output'>{escape(', '.join(stemmed_words))}</div>")
        card("📚 Lemmatization", f"<div class='token-output'>{escape(', '.join(lemmatized_words))}</div>")
        card("🏷 POS Tagging", "")
        st.dataframe(pd.DataFrame(pos_tags, columns=["Word", "POS"]), use_container_width=True, hide_index=True)
        card("🧠 Named Entity Recognition", "")
        st.dataframe(pd.DataFrame([(ent.text, ent.label_) for ent in doc.ents], columns=["Entity", "Label"]), use_container_width=True, hide_index=True)
        card("🔗 Dependency Parsing", "")
        st.dataframe(pd.DataFrame(dependencies, columns=["Word", "Dependency", "Head"]), use_container_width=True, hide_index=True)
        card("📦 Noun Phrase Chunking", "".join(f"<div class='token-output'>• {escape(chunk)}</div>" for chunk in noun_phrases) or "<div class='token-output'>No noun phrases found.</div>")

        report = f"""NLP TEXT ANALYZER REPORT\n{'=' * 28}\n\nOriginal Text:\n{text}\n\nSentence Segmentation:\n""" + "\n".join(f"{i}: {sent}" for i, sent in enumerate(sentences, 1)) + f"""\n\nWord Tokenization:\n{words}\n\nAfter Stop Word Removal:\n{filtered_words}\n\nAfter Stemming:\n{stemmed_words}\n\nAfter Lemmatization:\n{lemmatized_words}\n\nPOS Tagging:\n""" + "\n".join(f"{word} | {tag}" for word, tag in pos_tags) + "\n\nNamed Entity Recognition (NER):\n" + "\n".join(f"{ent.text} -> {ent.label_}" for ent in doc.ents) + "\n\nDependency Parsing:\n" + "\n".join(f"{word} -> {dependency} -> {head}" for word, dependency, head in dependencies) + "\n\nNoun Phrase Chunking:\n" + "\n".join(noun_phrases)


        st.download_button(
            label="📥 Download Report",
            data=report,
            file_name="NLP_Report.txt",
            mime="text/plain",
            use_container_width=True,
        )


st.markdown("<div class='footer'>Made with ❤️ using Streamlit, NLTK and spaCy</div>", unsafe_allow_html=True)
