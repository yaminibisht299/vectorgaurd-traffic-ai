import streamlit as st
import pandas as pd
import json
import os
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="VectorGuard - Live Violation Feed",
    page_icon="🚨",
    layout="wide"
)

# --- CLEAN HEADER STYLING ---
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .feed-header {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1.85rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="feed-header">🚨 Live Violation Feed</div>', unsafe_allow_html=True)

ALERTS_FILE = "traffic_alerts.jsonl"
REQUIRED_COLUMNS = [
    "Serial No.", 
    "Time", 
    "Camera ID", 
    "Vehicle ID", 
    "Violation Type", 
    "speed_kmh", 
    "speed_limit_kmh"
]

# Fixed single placeholder for the table (prevents duplication)
table_placeholder = st.empty()

def load_events():
    records = []
    if os.path.exists(ALERTS_FILE):
        with open(ALERTS_FILE, "r") as f:
            for idx, line in enumerate(f, start=1):
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        records.append({
                            "Serial No.": idx,
                            "Time": data.get("timestamp", "-"),
                            "Camera ID": data.get("camera_id", "-"),
                            "Vehicle ID": data.get("track_id", "-"),
                            "Violation Type": data.get("violation", "-"),
                            "speed_kmh": f"{float(data.get('speed_kmh')):.1f}" if data.get("speed_kmh") is not None else "None",
                            "speed_limit_kmh": str(data.get("speed_limit_kmh")) if data.get("speed_limit_kmh") is not None else "None"
                        })
                    except Exception:
                        pass
    return records

# --- CONTINUOUS TABLE REFRESH LOOP ---
while True:
    records = load_events()

    if records:
        df = pd.DataFrame(records)[REQUIRED_COLUMNS]
        # Show newest violations at the top
        display_df = df.iloc[::-1]
    else:
        # Default empty table waiting for detections
        display_df = pd.DataFrame(columns=REQUIRED_COLUMNS)

    # Render strictly in the fixed placeholder
    table_placeholder.dataframe(
        display_df,
        use_container_width=True,
        height=600,
        hide_index=True
    )

    time.sleep(1.0)