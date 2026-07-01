import streamlit as st
import requests

st.set_page_config(page_title="Medicine Recommendation System", layout="wide")
st.title("🩺 DiagnosAI")
st.markdown(
    """
    Predict diseases from symptoms, receive personalized medicine and health recommendations,
    and ask medical questions powered by AI and Retrieval-Augmented Generation (RAG).
    """
)

tab1, tab2 = st.tabs(["Disease prediction", "Medical assistant"])

with tab1:
    symptoms = st.text_area(
        "Please enter symptoms you would like to predict",
        placeholder="e.g. itching, skin_rash, chills, high_fever",
        help="Separate each symptom with a comma"
    )
    if st.button("Predict"):
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json={"symptoms": symptoms}
        )
        result = response.json()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader(result['disease'])
        with col2:
            st.metric("Severity score", result['severity_score'])

        st.write(result['description'])

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.subheader("Precautions")
            for item in result['precautions']:
                st.write(f"• {item}")
        with col2:
            st.subheader("Medications")
            for item in result['medications']:
                st.write(f"• {item}")
        with col3:
            st.subheader("Diets")
            for item in result['diets']:
                st.write(f"• {item}")

        with col4:
            st.subheader("Workouts")
            for item in result['workouts']:
                st.write(f"• {item}")

with tab2:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input("Ask a medical question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            st.write("🚧 RAG coming soon — medical knowledge assistant will be available here.")
        st.session_state.messages.append({"role": "assistant", "content": "🚧 RAG coming soon."})


