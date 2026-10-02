"""Face Re-ID: Streamlit demo UI (M3).

Run from the repo root:  streamlit run src/ui/app.py
"""
import sys
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.ui import backend  # noqa: E402

st.set_page_config(page_title="Face Re-ID", page_icon="🔍", layout="wide")

st.markdown("""
<style>
  .block-container {padding-top: 2rem; max-width: 1200px;}
  h1 {font-weight: 700; letter-spacing: -0.02em; margin-bottom: 0.2rem;}
  .sub {color: #5B6A70; margin-bottom: 1.5rem; font-size: 1.05rem;}
  .score {font-weight: 600; font-size: 0.95rem;}
  .bar {height: 6px; border-radius: 3px; background: #DDE3E1; overflow: hidden;}
  .bar > div {height: 100%; background: #1F4E5F;}
  .chip {display:inline-block; padding:2px 10px; border-radius:12px;
         background:#EEF1EF; color:#1F4E5F; font-size:0.8rem;}
  @media (prefers-reduced-motion: no-preference) {
    .stSpinner {transition: opacity .2s;}
  }
</style>
""", unsafe_allow_html=True)


def draw_boxes(img: Image.Image, boxes, selected=None) -> Image.Image:
    img = img.copy().convert("RGB")
    d = ImageDraw.Draw(img)
    w = max(2, img.width // 200)
    for i, b in enumerate(boxes):
        color = "#1F4E5F" if i == selected else "#E0A030"
        d.rectangle(b, outline=color, width=w)
        d.text((b[0] + 4, b[1] + 2), str(i + 1), fill=color)
    return img


def save_temp(uploaded) -> str:
    suffix = Path(uploaded.name).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(uploaded.getbuffer())
        return f.name


# ---------- header ----------
st.title("Face Re-ID")
st.markdown('<div class="sub">Upload a photo of a person to find every image '
            'in the database where they appear.</div>', unsafe_allow_html=True)

if backend.USING_MOCK:
    st.info("Running on mock data. The real search backend is not connected yet.")

# ---------- sidebar ----------
with st.sidebar:
    st.header("Search settings")
    threshold = st.slider(
        "Similarity threshold", 0.0, 1.0, 0.40, 0.01,
        help="Higher means stricter matching. Use the value calibrated on LFW.")
    top_k = st.number_input("Maximum results", 1, 100, 20)
    show_boxes = st.toggle("Highlight matched face", value=True)
    st.divider()
    st.caption("Limitations: accuracy drops on blurry, tiny or heavily angled "
               "faces. Similar-looking people may cross-match.")

# ---------- query input ----------
left, right = st.columns([1, 2], gap="large")

with left:
    uploaded = st.file_uploader("Query photo", type=["jpg", "jpeg", "png"])
    face_index = 0
    tmp_path = None

    if uploaded:
        tmp_path = save_temp(uploaded)
        img = Image.open(tmp_path).convert("RGB")
        try:
            with st.spinner("Detecting faces..."):
                faces = backend.detect_faces(tmp_path)
        except Exception as e:
            faces = []
            st.error(f"Face detection failed: {e}")

        if not faces:
            st.image(img, use_container_width=True)
            if not st.session_state.get("_err"):
                st.warning("No face found in this photo. Try a clearer, "
                           "front-facing photo.")
        else:
            if len(faces) > 1:
                st.write(f"{len(faces)} faces found. Choose who to search for.")
                face_index = st.radio(
                    "Face", range(len(faces)), horizontal=True,
                    format_func=lambda i: f"Face {i + 1}", label_visibility="collapsed")
            st.image(draw_boxes(img, [f["bbox"] for f in faces], face_index),
                     use_container_width=True)
    search = st.button("Search", type="primary", disabled=not (uploaded and tmp_path),
                       use_container_width=True)

# ---------- results ----------
with right:
    if not uploaded:
        st.subheader("Results")
        st.write("Upload a query photo on the left, then select Search.")
    elif search:
        try:
            with st.spinner("Searching the database..."):
                results = backend.query(tmp_path, face_index=face_index,
                                        threshold=threshold, top_k=int(top_k))
            st.session_state["results"] = results
            st.session_state["ran_threshold"] = threshold
            st.session_state.pop("_err", None)
        except Exception as e:
            st.session_state["results"] = None
            st.session_state["_err"] = str(e)

    if uploaded and st.session_state.get("_err"):
        st.error(f"Search failed: {st.session_state['_err']}")
    elif uploaded and st.session_state.get("results") is not None:
        results = st.session_state["results"]
        t = st.session_state.get("ran_threshold", threshold)
        if not results:
            st.subheader("No match found")
            st.write(f"No image has a face scoring above {t:.2f}. "
                     "Lower the threshold in the sidebar or try another photo.")
        else:
            st.subheader(f"{len(results)} matching image"
                         f"{'s' if len(results) != 1 else ''}")
            st.caption(f"Threshold {t:.2f}, sorted by similarity.")
            cols = st.columns(3)
            for i, r in enumerate(results):
                with cols[i % 3]:
                    try:
                        im = backend.load_image(r["image_id"])
                        if show_boxes and r.get("bbox"):
                            im = draw_boxes(im, [r["bbox"]], selected=0)
                        st.image(im, use_container_width=True)
                    except Exception:
                        st.warning(f"Could not load {r['image_id']}")
                    pct = max(0.0, min(1.0, r["score"])) * 100
                    st.markdown(
                        f'<div class="score">{r["score"]:.3f}</div>'
                        f'<div class="bar"><div style="width:{pct:.0f}%"></div></div>'
                        f'<div style="color:#5B6A70;font-size:.8rem;margin:4px 0 14px">'
                        f'{r["image_id"]}</div>', unsafe_allow_html=True)
