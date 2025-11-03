import streamlit as st
import pandas as pd
import json
from collections import Counter

st.set_page_config(page_title="Power Ops Insights", layout="wide")

st.title("AI-Driven Fault Analysis")

uploaded = st.file_uploader("Upload JSONL (instruction or fault schema)", type=["jsonl"]) 

if uploaded is not None:
    lines = uploaded.read().decode("utf-8").splitlines()
    rows = [json.loads(l) for l in lines if l.strip()]
    st.success(f"Loaded {len(rows)} records")

    if rows and isinstance(rows[0], dict):
        df = pd.DataFrame(rows)
        st.dataframe(df.head(20), use_container_width=True)

        if "fault_type" in df.columns:
            counts = Counter(df["fault_type"].dropna())
            st.subheader("Fault Type Frequency")
            st.bar_chart(pd.DataFrame({"fault_type": list(counts.keys()), "count": list(counts.values())}).set_index("fault_type"))

        if "labels" in df.columns:
            all_labels = [lab for labs in df["labels"].dropna() for lab in (labs or [])]
            label_counts = Counter(all_labels)
            st.subheader("Label Frequency")
            st.bar_chart(pd.DataFrame({"label": list(label_counts.keys()), "count": list(label_counts.values())}).set_index("label"))

st.info("Train a LoRA model with scripts/train_lora.py and point the app to generated insights later.")
