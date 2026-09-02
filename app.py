import os
import re
import json
import pickle
import tempfile

import numpy as np
import faiss
import torch
import torch.nn as nn
import whisper
import streamlit as st
from PIL import Image
from transformers import (
    AutoProcessor,
    CLIPModel,
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_BASE_PATH = os.path.join(BASE_DIR, "dataset", "knowledge_base", "campus_database.json")
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "saved_models", "faiss_index", "campus.index")
LABEL_MAPPING_PATH = os.path.join(BASE_DIR, "saved_models", "faiss_index", "location_mapping.pkl")
DISTILBERT_PATH = os.path.join(BASE_DIR, "saved_models", "distilbert")
FUSION_MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "fusion", "fusion_model.pt")
IMAGES_PATH = os.path.join(BASE_DIR, "dataset", "images")

LABEL2ID = {"ask_direction": 0, "ask_hours": 1, "find_event": 2, "find_location": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

INTENT_DISPLAY = {
    "find_location": "Find Location",
    "ask_hours": "Opening Hours",
    "find_event": "Find Event",
    "ask_direction": "Get Directions",
}


class FusionMLP(nn.Module):
    def __init__(self, image_dim=512, text_dim=768, num_classes=19):
        super().__init__()
        self.image_proj = nn.Linear(image_dim, 256)
        self.text_proj = nn.Linear(text_dim, 256)
        self.fusion = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, image_emb=None, text_emb=None):
        if image_emb is None:
            image_emb = torch.zeros(text_emb.size(0), 256)
        else:
            image_emb = torch.relu(self.image_proj(image_emb))
        if text_emb is None:
            text_emb = torch.zeros(image_emb.size(0), 256)
        else:
            text_emb = torch.relu(self.text_proj(text_emb))
        combined = torch.cat([image_emb, text_emb], dim=1)
        return self.fusion(combined)


@st.cache_resource(show_spinner="Loading CLIP model...")
def load_clip():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, processor


@st.cache_resource(show_spinner="Loading Whisper model...")
def load_whisper():
    model = whisper.load_model("base")
    return model


@st.cache_resource(show_spinner="Loading DistilBERT intent classifier...")
def load_distilbert():
    tokenizer = DistilBertTokenizerFast.from_pretrained(DISTILBERT_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(DISTILBERT_PATH)
    model.eval()
    return tokenizer, model


@st.cache_resource(show_spinner="Loading FAISS index...")
def load_faiss():
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(LABEL_MAPPING_PATH, "rb") as f:
        labels = pickle.load(f)
    return index, labels


@st.cache_data(show_spinner="Loading campus knowledge base...")
def load_knowledge_base():
    with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


STOPWORDS = {"a", "an", "the", "is", "in", "at", "of", "to", "and", "or", "for", "on", "are", "was", "it", "with"}

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)


def query_kb(kb, intent_label, entity=None):
    if entity:
        entity_lower = entity.lower().replace("_", " ")
    else:
        entity_lower = None

    for record in kb:
        name = record.get("building_name", "").lower()
        category = record.get("category", "").lower()
        folder = record.get("folder_name", "").lower()
        description = record.get("description", "").lower()

        if entity_lower:
            if (entity_lower in name
                    or entity_lower in category
                    or entity_lower in folder
                    or entity_lower == folder
                    or entity_lower in description):
                return record

    if intent_label == "find_event":
        for record in kb:
            events = record.get("events", "")
            if events and events.lower() != "no scheduled events":
                return record

    return kb[0] if kb else {}


def extract_entity_from_text(text: str) -> str | None:
    text_lower = text.lower()
    kb = load_knowledge_base()
    best_match = None
    for record in kb:
        for field in ["folder_name", "building_name", "category"]:
            value = record.get(field, "").lower()
            if value and value in text_lower:
                if best_match is None or len(value) > len(best_match):
                    best_match = value
    return best_match


def process_image(image: Image.Image):
    clip_model, clip_processor = load_clip()
    index, faiss_labels = load_faiss()

    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        output = clip_model.vision_model(**inputs)
    vec = output.pooler_output.detach().cpu().numpy()[0].astype("float32")
    vec = vec / np.linalg.norm(vec)

    distances, indices = index.search(vec.reshape(1, -1), 3)
    top1_label = faiss_labels[indices[0][0]]
    top3_labels = [faiss_labels[idx] for idx in indices[0]]
    top3_scores = distances[0].tolist()

    return top1_label, top3_labels, top3_scores


def process_audio(audio_path: str) -> str:
    whisper_model = load_whisper()
    result = whisper_model.transcribe(audio_path)
    return result["text"].strip()


def classify_intent(text: str):
    tokenizer, model = load_distilbert()
    cleaned = clean_text(text)
    enc = tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=1)[0]
    intent_id = torch.argmax(probs).item()
    intent_label = ID2LABEL[intent_id]
    confidence = probs[intent_id].item()
    return intent_label, confidence


def get_sample_image(folder_name: str) -> Image.Image | None:
    folder_path = os.path.join(IMAGES_PATH, folder_name)
    if os.path.isdir(folder_path):
        files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if files:
            return Image.open(os.path.join(folder_path, files[0])).convert("RGB")
    return None


st.set_page_config(
    page_title="Smart Campus Assistant",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, .stApp {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(48, 43, 99, 0.35);
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: rgba(255, 255, 255, 0.75);
        font-size: 1.05rem;
        font-weight: 400;
        margin: 0;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
        font-size: 0.95rem;
    }

    .result-card {
        background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.2);
    }
    .result-card h2 {
        color: #a78bfa;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0 0 0.3rem 0;
    }
    .result-card .category-badge {
        display: inline-block;
        background: rgba(99, 102, 241, 0.2);
        color: #818cf8;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 1rem;
        letter-spacing: 0.3px;
    }
    .result-card .info-row {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        padding: 10px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        color: #e2e8f0;
        font-size: 0.95rem;
    }
    .result-card .info-row:last-child {
        border-bottom: none;
    }
    .result-card .info-label {
        font-weight: 600;
        color: #94a3b8;
        min-width: 110px;
    }
    .result-card .info-value {
        color: #f1f5f9;
    }

    .pipeline-box {
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 0.5rem 0 1rem 0;
    }
    .pipeline-box h4 {
        color: #38bdf8;
        font-size: 0.9rem;
        font-weight: 700;
        margin: 0 0 0.6rem 0;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    .pipeline-box p {
        color: #94a3b8;
        font-size: 0.88rem;
        margin: 3px 0;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e2e8f0;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li {
        color: #94a3b8;
        font-size: 0.9rem;
    }

    .metric-row {
        display: flex;
        gap: 12px;
        margin: 1rem 0;
    }
    .metric-card {
        flex: 1;
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-card .metric-value {
        color: #a78bfa;
        font-size: 1.6rem;
        font-weight: 800;
    }
    .metric-card .metric-label {
        color: #64748b;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## Campus Assistant")
    st.markdown("---")
    st.markdown("### How to Use")
    st.markdown("""
    1. **Image** - Upload a photo of a campus building or sign
    2. **Voice** - Upload an audio recording of your question
    3. **Text** - Type your question directly
    """)
    st.markdown("---")
    st.markdown("### Model Pipeline")
    st.markdown("""
    - **CLIP + FAISS** - Image retrieval
    - **Whisper** - Speech transcription
    - **DistilBERT** - Intent classification
    - **Fusion MLP** - Multimodal fusion
    - **Knowledge Base** - Campus information
    """)
    st.markdown("---")
    st.markdown("### Evaluation Metrics")
    st.markdown("""
    | Metric | Score |
    |---|---|
    | CLIP Top-1 | 81.1% |
    | CLIP Top-3 | 95.8% |
    | Intent F1 | 100% |
    | KB Retrieval | 60.0% |
    """)
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:#64748b; font-size:0.75rem;'>"
        "Smart Campus AI Assistant v1.0<br>BSBI Campus Orientation</p>",
        unsafe_allow_html=True,
    )

st.markdown("""
<div class="main-header">
    <h1>Smart Campus Multimodal AI Assistant</h1>
    <p>Navigate your campus with AI - upload an image, speak your question, or type it in</p>
</div>
""", unsafe_allow_html=True)


def render_result(record, intent_label=None, modalities_used=None, confidence=None, transcript=None, image_match=None):
    building = record.get("building_name", "Unknown")
    category = record.get("category", "N/A")
    description = record.get("description", "No description available.")
    hours = record.get("opening_hours", "N/A")
    map_ref = record.get("map_reference", "N/A")
    events = record.get("events", "No scheduled events")
    folder = record.get("folder_name", "")

    if intent_label:
        intent_name = INTENT_DISPLAY.get(intent_label, intent_label)
    else:
        intent_name = "Location Match"

    st.markdown(f"""
    <div class="result-card">
        <h2>{building}</h2>
        <span class="category-badge">{category}</span>
        <div class="info-row">
            <span class="info-label">Description</span>
            <span class="info-value">{description}</span>
        </div>
        <div class="info-row">
            <span class="info-label">Opening Hours</span>
            <span class="info-value">{hours}</span>
        </div>
        <div class="info-row">
            <span class="info-label">Directions</span>
            <span class="info-value">{map_ref}</span>
        </div>
        <div class="info-row">
            <span class="info-label">Events</span>
            <span class="info-value">{events}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    pipeline_parts = []
    if modalities_used:
        pipeline_parts.append(f"<p><b>Modalities:</b> {', '.join(modalities_used)}</p>")
    if intent_label:
        conf_str = f" ({confidence*100:.1f}%)" if confidence else ""
        pipeline_parts.append(f"<p><b>Intent:</b> {intent_name}{conf_str}</p>")
    if transcript:
        pipeline_parts.append(f"<p><b>Transcript:</b> \"{transcript}\"</p>")
    if image_match:
        pipeline_parts.append(f"<p><b>Image Match:</b> {image_match}</p>")

    if pipeline_parts:
        st.markdown(f"""
        <div class="pipeline-box">
            <h4>Pipeline Details</h4>
            {''.join(pipeline_parts)}
        </div>
        """, unsafe_allow_html=True)

    sample_img = get_sample_image(folder)
    if sample_img:
        st.image(sample_img, caption=f"Reference image - {building}", use_container_width=True)


tab_image, tab_voice, tab_text = st.tabs(["Image Input", "Voice Input", "Text Input"])

with tab_image:
    st.markdown("### Upload a Campus Photo")
    st.markdown("Upload a photograph of a campus building, sign, or indoor area to identify the location.")

    uploaded_image = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="image_uploader",
        help="Supported formats: JPG, JPEG, PNG, BMP, WEBP",
    )

    col_img_left, col_img_right = st.columns([1, 1])

    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        with col_img_left:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        with col_img_right:
            with st.spinner("Analyzing image with CLIP + FAISS..."):
                top1, top3, scores = process_image(image)
                kb = load_knowledge_base()
                record = query_kb(kb, "find_location", top1)

            st.markdown("#### Top-3 Matches")
            for i, (label, score) in enumerate(zip(top3, scores)):
                rank = ["1st", "2nd", "3rd"][i]
                st.markdown(f"{rank}: **{label.replace('_', ' ').title()}** - similarity: `{score:.4f}`")

        st.markdown("---")
        render_result(
            record,
            intent_label="find_location",
            modalities_used=["Image (CLIP + FAISS)"],
            image_match=top1.replace("_", " ").title(),
        )

with tab_voice:
    st.markdown("### Upload a Voice Query")
    st.markdown("Upload an audio recording of your campus question (e.g., *\"Where is the library?\"*).")

    uploaded_audio = st.file_uploader(
        "Choose an audio file",
        type=["wav", "mp3", "m4a", "ogg", "flac"],
        key="audio_uploader",
        help="Supported formats: WAV, MP3, M4A, OGG, FLAC",
    )

    if uploaded_audio is not None:
        st.audio(uploaded_audio, format="audio/wav")

        with st.spinner("Transcribing with Whisper..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(uploaded_audio.getvalue())
                tmp_path = tmp.name
            transcript = process_audio(tmp_path)
            os.unlink(tmp_path)

        st.success(f'Transcript: "{transcript}"')

        with st.spinner("Classifying intent with DistilBERT..."):
            intent_label, confidence = classify_intent(transcript)
            entity = extract_entity_from_text(transcript)
            kb = load_knowledge_base()
            record = query_kb(kb, intent_label, entity)

        render_result(
            record,
            intent_label=intent_label,
            modalities_used=["Voice (Whisper)", "Text (DistilBERT)"],
            confidence=confidence,
            transcript=transcript,
        )

with tab_text:
    st.markdown("### Type Your Question")
    st.markdown("Ask anything about campus locations, opening hours, events, or directions.")

    st.markdown("**Try these examples:**")
    example_cols = st.columns(4)
    examples = [
        "Where is the library?",
        "What time does the cafeteria open?",
        "Are there events at the auditorium?",
        "How do I get to the gym?",
    ]
    for i, (col, example) in enumerate(zip(example_cols, examples)):
        with col:
            if st.button(example, key=f"example_{i}", use_container_width=True):
                st.session_state["text_query"] = example

    text_query = st.text_input(
        "Your question:",
        value=st.session_state.get("text_query", ""),
        placeholder="e.g., Where can I find the computer lab?",
        key="text_input_field",
    )

    if st.button("Search", key="text_search_btn", type="primary", use_container_width=True):
        if text_query.strip():
            with st.spinner("Processing your query..."):
                intent_label, confidence = classify_intent(text_query)
                entity = extract_entity_from_text(text_query)
                kb = load_knowledge_base()
                record = query_kb(kb, intent_label, entity)

            render_result(
                record,
                intent_label=intent_label,
                modalities_used=["Text (DistilBERT)"],
                confidence=confidence,
            )
        else:
            st.warning("Please enter a question to search.")

st.markdown("---")
with st.expander("Browse Full Campus Knowledge Base", expanded=False):
    kb = load_knowledge_base()
    st.markdown(f"**{len(kb)} campus locations** in the knowledge base:")

    table_data = []
    for rec in kb:
        table_data.append({
            "Building": rec.get("building_name", ""),
            "Category": rec.get("category", ""),
            "Hours": rec.get("opening_hours", ""),
            "Location": rec.get("map_reference", ""),
            "Events": rec.get("events", ""),
        })
    st.dataframe(table_data, use_container_width=True, hide_index=True)
