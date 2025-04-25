import streamlit as st

st.set_page_config(page_title="BYOB EMA Dashboard", layout="wide")

st.title("🚀 BYOB EMA Dashboard")
st.markdown("Welcome to your 0DTE strategy dashboard! More visualizations coming soon.")

# Add a simple slider and chart just to see it render
value = st.slider("Choose number of entries", 2, 20, 10)
st.write(f"You selected **{value}** entries")

# test edit