import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Material Palette RAG", layout="wide")
st.title("Apparel Material Palette RAG")

backend_url = st.sidebar.text_input("Backend URL", value=os.getenv("BACKEND_URL", "http://localhost:8000"))
top_k = st.sidebar.slider("Top K (retrieved rows)", 3, 12, 6, 1)

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask about materials, content, benefits, supplier, season usage, etc.")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching and generating..."):
            try:
                resp = requests.post(
                    f"{backend_url.rstrip('/')}/chat",
                    json={"message": prompt, "top_k": int(top_k)},
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
                st.markdown(data["answer"])

                with st.expander("Retrieved rows (citations)"):
                    for c in data.get("citations", []):
                        st.write(f"row_id={c['row_id']}  score={c['score']:.4f}")
                        if c.get("material"):
                            st.write(c["material"])
                        if c.get("snippet"):
                            st.caption(c["snippet"])
                        st.divider()

                st.session_state.messages.append({"role": "assistant", "content": data["answer"]})
            except Exception as e:
                st.error(f"Error: {e}")
