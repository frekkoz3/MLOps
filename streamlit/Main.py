import streamlit as st

# -------------------------------
# 1. Streamlit App Configuration
# -------------------------------
st.set_page_config(
    page_title="440MI - Machine Learning Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 Machine Learning Streamlit Hub")
st.markdown("""
Welcome to the **440MI + 305SM Course**.  
This interface automatically discovers all available modules inside the `pages/` directory and allows you to navigate between them.  
Each page represents a distinct component — such as model training, online learning, visualization, or drift monitoring.
""")


# -------------------------------
# 4. Footer
# -------------------------------
st.markdown("---")
st.markdown(
    "<small>Developed for 440MI – University of Trieste | Modular ML Dashboard using Streamlit.</small>",
    unsafe_allow_html=True,
)
