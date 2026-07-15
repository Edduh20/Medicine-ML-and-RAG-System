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

VALID_SYMPTOMS = [
    'abdominal_pain', 'acidity', 'acute_liver_failure', 'altered_sensorium', 'anxiety',
    'back_pain', 'blackheads', 'bladder_discomfort', 'blister', 'bloody_stool',
    'blurred_and_distorted_vision', 'breathlessness', 'bruising', 'burning_micturition',
    'chest_pain', 'chills', 'cold_hands_and_feets', 'coma', 'congestion',
    'constipation', 'continuous_feel_of_urine', 'continuous_sneezing', 'cough',
    'cramps', 'dark_urine', 'dehydration', 'depression', 'diarrhoea',
    'dischromic_patches', 'distention_of_abdomen', 'dizziness', 'drying_and_tingling_lips',
    'enlarged_thyroid', 'excessive_hunger', 'extra_marital_contacts', 'family_history',
    'fast_heart_rate', 'fatigue', 'fluid_overload', 'fluid_overload.1', 'foul_smell_of urine',
    'headache', 'high_fever', 'hip_joint_pain', 'history_of_alcohol_consumption',
    'increased_appetite', 'indigestion', 'inflammatory_nails', 'internal_itching',
    'irregular_sugar_level', 'irritability', 'irritation_in_anus', 'itching',
    'joint_pain', 'knee_pain', 'lack_of_concentration', 'lethargy', 'loss_of_appetite',
    'loss_of_balance', 'loss_of_smell', 'malaise', 'mild_fever', 'mood_swings',
    'movement_stiffness', 'mucoid_sputum', 'muscle_pain', 'muscle_wasting',
    'muscle_weakness', 'nausea', 'neck_pain', 'nodal_skin_eruptions', 'obesity',
    'pain_during_bowel_movements', 'pain_in_anal_region', 'pain_behind_the_eyes',
    'painful_walking', 'palpitations', 'passage_of_gases', 'patches_in_throat',
    'phlegm', 'polyuria', 'prominent_veins_on_calf', 'puffy_face_and_eyes',
    'pus_filled_pimples', 'receiving_blood_transfusion', 'receiving_unsterile_injections',
    'red_sore_around_nose', 'red_spots_over_body', 'redness_of_eyes', 'restlessness',
    'runny_nose', 'rusty_sputum', 'scurring', 'shivering', 'silver_like_dusting',
    'sinus_pressure', 'skin_peeling', 'skin_rash', 'slurred_speech', 'small_dents_in_nails',
    'spinning_movements', 'spotting_urination', 'stiff_neck', 'stomach_bleeding',
    'stomach_pain', 'sunken_eyes', 'sweating', 'swelled_lymph_nodes', 'swelling_joints',
    'swelling_of_stomach', 'swollen_blood_vessels', 'swollen_extremeties', 'swollen_legs',
    'throat_irritation', 'toxic_look_(typhos)', 'ulcers_on_tongue', 'unsteadiness',
    'visual_disturbances', 'vomiting', 'watering_from_eyes', 'weakness_in_limbs',
    'weakness_of_one_body_side', 'weight_gain', 'weight_loss', 'yellow_crust_ooze',
    'yellow_urine', 'yellowing_of_eyes', 'yellowish_skin'
]

with tab1:
    select_symptoms = st.multiselect(
        "Select your symptoms",
        options=VALID_SYMPTOMS,
        help="Select one or more symptoms from the list to predict the disease"
    )

    if st.button("Predict"):
        if not select_symptoms:
            st.warning("Please select at least one symptom.")
        else:
            symptoms = ",".join(select_symptoms)
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
                st.write(f"• {item.strip().capitalize()}")
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
    if "user_query" not in st.session_state:
        st.session_state.user_query = ""
    if "active_suggestion" not in st.session_state:
        st.session_state.active_suggestion = None


    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if st.session_state.active_suggestion:
        st.write("---")
        if st.button(f"💬 Ask: {st.session_state.active_suggestion}"):
            st.session_state.user_query = st.session_state.active_suggestion
            st.session_state.active_suggestion = None
            st.rerun()

    user_input = st.chat_input("Ask a medical question...")

    prompt = user_input or (st.session_state.user_query if st.session_state.user_query else None)


    if st.session_state.user_query:
        st.session_state.user_query = ""


    if prompt:
        st.session_state.active_suggestion = None

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(
                        "http://127.0.0.1:8000/answer_question",
                        json={"question": prompt}
                    )
                    result = response.json()
                    response_text = result["answer"]

                    if "Suggestion" in response_text:
                        main_answer, suggestion_part = response_text.split("Suggestion:")
                        main_answer = main_answer.strip()
                        suggested_question = suggestion_part.replace("[", "").replace("]", "").strip()

                        st.write(main_answer)

                        st.session_state.messages.append({"role": "assistant", "content": main_answer})

                        st.session_state.active_suggestion = suggested_question
                    else:
                        st.write(response_text)
                        st.session_state.messages.append({"role": "assistant", "content": response_text})
                        st.session_state.active_suggestion = None

                    st.rerun()

                except Exception as e:
                    st.error(f"Failed to connect to backend: {e}")

