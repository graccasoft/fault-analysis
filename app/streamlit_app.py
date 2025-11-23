import streamlit as st
import pandas as pd
import json
from collections import Counter
import os
from pathlib import Path
import sys
from datetime import datetime
import time
import plotly.express as px
import plotly.graph_objects as go

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

# Initialize query log file
QUERY_LOG_FILE = ROOT / "outputs" / "query_log.jsonl"
QUERY_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

def log_query(input_text, output_text, inference_time, instruction):
    """Log query to JSONL file"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "input": input_text,
        "output": output_text,
        "instruction": instruction,
        "input_length": len(input_text),
        "output_length": len(output_text),
        "inference_time": inference_time
    }
    with open(QUERY_LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

def load_query_logs():
    """Load query logs from JSONL file"""
    if not QUERY_LOG_FILE.exists():
        return pd.DataFrame()
    
    logs = []
    with open(QUERY_LOG_FILE, "r") as f:
        for line in f:
            if line.strip():
                logs.append(json.loads(line))
    
    if logs:
        df = pd.DataFrame(logs)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    return pd.DataFrame()

# Create tabs
tab1, tab2, tab3 = st.tabs(["🔍 Inference", "📊 Data Explorer", "📈 Statistics"])

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
                from transformers import AutoTokenizer
                from peft import AutoPeftModelForSeq2SeqLM
                
                # Load LoRA fine-tuned model with adapters
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoPeftModelForSeq2SeqLM.from_pretrained(model_path)
                
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
                    value="Analyze the power system event and generate a full technical assessment.",
                    help="The task instruction for the model"
                )
            
            fault_input = st.text_area(
                "Fault Description",
                height=150,
                placeholder="Example: Transformer T3 oil temperature reached 92°C. One cooling fan failed to start. Load at 78%. Slight humming noise noted. No protective trip.",
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
                            
                            # Generate with timing
                            start_time = time.time()
                            outputs = model.generate(
                                **inputs,
                                max_length=max_length,
                                temperature=temperature,
                                do_sample=True,
                                top_p=0.95,
                                num_return_sequences=1,
                                repetition_penalty=1.2,
                                length_penalty=1.0,
                                early_stopping=True
                            )
                            inference_time = time.time() - start_time
                            
                            # Decode
                            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
                            
                            # Log the query
                            log_query(fault_input, result, inference_time, instruction)
                            
                            # Display result
                            st.subheader("Analysis Result")
                            st.success(result)
                            st.caption(f"⏱️ Inference time: {inference_time:.2f}s")
                            
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
                    "Transformer T3 oil temperature reached 92°C. One cooling fan failed to start. Load at 78%. Slight humming noise noted. No protective trip.",
                    "TX-4 winding temperature exceeded 105°C. Cooling system active but ineffective. Load is 95% of rated capacity. Oil leakage observed.",
                    "Capacitor bank CB-2 reported unbalanced currents. One capacitor unit failed open. Excessive harmonic distortion detected in the feeder."
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

# Tab 3: Statistics
with tab3:
    st.header("Query Statistics")
    
    df_logs = load_query_logs()
    
    if df_logs.empty:
        st.info("📊 No queries logged yet. Start analyzing faults to see statistics here!")
    else:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Queries", len(df_logs))
        with col2:
            st.metric("Avg Inference Time", f"{df_logs['inference_time'].mean():.2f}s")
        with col3:
            st.metric("Avg Input Length", f"{df_logs['input_length'].mean():.0f} chars")
        with col4:
            st.metric("Avg Output Length", f"{df_logs['output_length'].mean():.0f} chars")
        
        st.divider()
        
        # Visualizations
        col_left, col_right = st.columns(2)
        
        with col_left:
            # Queries over time
            st.subheader("📅 Queries Over Time")
            df_logs['date'] = df_logs['timestamp'].dt.date
            queries_per_day = df_logs.groupby('date').size().reset_index(name='count')
            fig_timeline = px.line(
                queries_per_day, 
                x='date', 
                y='count',
                labels={'date': 'Date', 'count': 'Number of Queries'},
                markers=True
            )
            fig_timeline.update_layout(height=300)
            st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Inference time distribution
            st.subheader("⏱️ Inference Time Distribution")
            fig_time = px.histogram(
                df_logs, 
                x='inference_time',
                nbins=20,
                labels={'inference_time': 'Inference Time (seconds)'},
            )
            fig_time.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_time, use_container_width=True)
        
        with col_right:
            # Input vs Output length scatter
            st.subheader("📏 Input vs Output Length")
            fig_scatter = px.scatter(
                df_logs,
                x='input_length',
                y='output_length',
                color='inference_time',
                labels={
                    'input_length': 'Input Length (chars)',
                    'output_length': 'Output Length (chars)',
                    'inference_time': 'Time (s)'
                },
                color_continuous_scale='Viridis'
            )
            fig_scatter.update_layout(height=300)
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Recent queries table
            st.subheader("🕐 Recent Queries")
            recent_df = df_logs.sort_values('timestamp', ascending=False).head(5)[[
                'timestamp', 'input_length', 'output_length', 'inference_time'
            ]].copy()
            recent_df['timestamp'] = recent_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            recent_df.columns = ['Time', 'Input Len', 'Output Len', 'Inference (s)']
            st.dataframe(recent_df, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # Detailed query log
        with st.expander("🔍 View All Query Details"):
            display_df = df_logs.sort_values('timestamp', ascending=False)[[
                'timestamp', 'input', 'output', 'inference_time'
            ]].copy()
            display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            display_df.columns = ['Timestamp', 'Input', 'Output', 'Inference Time (s)']
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Download logs
        if st.button("📥 Download Query Logs (CSV)"):
            csv = df_logs.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"query_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
