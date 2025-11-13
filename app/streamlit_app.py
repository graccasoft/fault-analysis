import streamlit as st
import pandas as pd
import json
from collections import Counter
import os
from pathlib import Path
import sys

# Add src to path for imports
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

st.set_page_config(page_title="Power Ops Insights..", layout="wide")

st.title("AI-Driven Fault Analysis")

# Sidebar for model selection
with st.sidebar:
    st.header("Model Configuration")
    model_dir = st.text_input(
        "Model Directory",
        value="outputs/flan_t5_lora",
        help="Path to the trained model directory"
    )
    
    model_exists = os.path.exists(model_dir) and os.path.isdir(model_dir)
    if model_exists:
        st.success("✓ Model found")
    else:
        st.warning("⚠ Model not found. Train a model first.")

# Create tabs
tab1, tab2 = st.tabs(["🔍 Inference", "📊 Data Explorer"])

# Tab 1: Inference
with tab1:
    st.header("Fault Analysis Inference")
    
    if not model_exists:
        st.error("No trained model found. Please train a model first using:")
        st.code("docker-compose --profile training run --rm trainer")
        st.info("Or locally: `python3 scripts/train_lora.py --config configs/config.example.yaml`")
    else:
        # Load model (cached)
        @st.cache_resource
        def load_model(model_path):
            try:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
                
                # Load base model and tokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
                
                return model, tokenizer
            except Exception as e:
                st.error(f"Error loading model: {e}")
                return None, None
        
        model, tokenizer = load_model(model_dir)
        
        if model is not None and tokenizer is not None:
            st.success("✅ Model loaded successfully!")
            
            # Input section
            st.subheader("Enter Fault Description")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                instruction = st.text_input(
                    "Instruction",
                    value="Classify the likely fault and suggest an action.",
                    help="The task instruction for the model"
                )
            
            fault_input = st.text_area(
                "Fault Description",
                height=150,
                placeholder="Example: Feeder A experienced intermittent undervoltage alarms. Field team reported vegetation near line in wet conditions.",
                help="Describe the fault or incident"
            )
            
            with col2:
                max_length = st.slider("Max Output Length", 50, 512, 256)
                temperature = st.slider("Temperature", 0.1, 2.0, 0.7, 0.1)
            
            if st.button("🔍 Analyze Fault", type="primary", use_container_width=True):
                if fault_input.strip():
                    with st.spinner("Analyzing..."):
                        try:
                            # Format input
                            from fault_analysis.preprocess.clean import normalize_text
                            formatted_input = f"Instruction: {instruction}\nInput: {normalize_text(fault_input)}\nAnswer:"
                            
                            # Tokenize
                            inputs = tokenizer(
                                formatted_input,
                                return_tensors="pt",
                                max_length=512,
                                truncation=True
                            )
                            
                            # Generate
                            outputs = model.generate(
                                **inputs,
                                max_length=max_length,
                                temperature=temperature,
                                do_sample=True,
                                top_p=0.95,
                                num_return_sequences=1
                            )
                            
                            # Decode
                            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
                            
                            # Display result
                            st.subheader("Analysis Result")
                            st.success(result)
                            
                            # Show formatted output
                            with st.expander("View Details"):
                                st.json({
                                    "instruction": instruction,
                                    "input": fault_input,
                                    "output": result,
                                    "model": model_dir
                                })
                        
                        except Exception as e:
                            st.error(f"Error during inference: {e}")
                            import traceback
                            st.code(traceback.format_exc())
                else:
                    st.warning("Please enter a fault description.")
            
            # Example prompts
            with st.expander("📝 Example Prompts"):
                examples = [
                    "Feeder A experienced intermittent undervoltage alarms. Field team reported vegetation near line in wet conditions.",
                    "Transformer T3 oil temp reached 95C during evening peak. Cooling fan 2 reported failure.",
                    "Circuit breaker CB-12 tripped three times in 24 hours. No visible damage observed."
                ]
                for i, ex in enumerate(examples, 1):
                    st.markdown(f"**Example {i}:** {ex}")

# Tab 2: Data Explorer
with tab2:
    st.header("Data Explorer")
    
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
    else:
        st.info("Upload a JSONL file to explore training data statistics.")
    
    st.divider()
    st.info("💡 Train a LoRA model with: `docker-compose --profile training run --rm trainer`")
