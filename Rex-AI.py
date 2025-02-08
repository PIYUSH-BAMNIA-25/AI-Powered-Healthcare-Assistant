import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForTokenClassification

# Load BioBERT for Question Answering
QA_MODEL_NAME = "dmis-lab/biobert-v1.1"
qa_tokenizer = AutoTokenizer.from_pretrained(QA_MODEL_NAME)
qa_model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL_NAME)
qa_pipeline = pipeline("question-answering", model=qa_model, tokenizer=qa_tokenizer)

# Load BioBERT for Named Entity Recognition (NER)
NER_MODEL_NAME = "dmis-lab/biobert-v1.1"
ner_tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_NAME)
ner_model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_NAME)
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer)

# Load Medical Data (Manually Defined or Load from JSON)
medical_data = {
    "cold": {
        "symptoms": ["sneezing", "runny nose", "sore throat", "headache"],
        "medication": "Paracetamol, Antihistamines, Rest, Hydration",
        "diet": "Warm fluids, Ginger tea, Honey, Vitamin C-rich foods",
        "meditation": "Deep breathing exercises, Relaxation techniques"
    },
    "migraine": {
        "symptoms": ["severe headache", "nausea", "sensitivity to light and sound"],
        "medication": "Ibuprofen, Naproxen, Triptans",
        "diet": "Magnesium-rich foods, Avoid caffeine and alcohol",
        "meditation": "Mindfulness, Yoga, Gentle stretching"
    },
    "flu": {
        "symptoms": ["fever", "chills", "muscle aches", "cough", "fatigue"],
        "medication": "Antiviral drugs, Pain relievers, Rest, Hydration",
        "diet": "Chicken soup, Herbal teas, Fruits, Vegetables",
        "meditation": "Guided imagery, Progressive muscle relaxation"
    },
    "diabetes": {
        "symptoms": ["increased thirst", "frequent urination", "fatigue", "blurred vision"],
        "medication": "Metformin, Insulin therapy, Sulfonylureas",
        "diet": "Low-carb diet, High-fiber foods, Lean proteins",
        "meditation": "Breathing exercises, Stress management techniques"
    },
    "hypertension": {
        "symptoms": ["high blood pressure", "dizziness", "chest pain", "shortness of breath"],
        "medication": "ACE inhibitors, Beta-blockers, Calcium channel blockers",
        "diet": "Low-sodium diet, Green leafy vegetables, Bananas, Oatmeal",
        "meditation": "Progressive muscle relaxation, Meditation, Deep breathing"
    },
    "asthma": {
        "symptoms": ["shortness of breath", "wheezing", "chest tightness", "coughing"],
        "medication": "Inhalers, Bronchodilators, Corticosteroids",
        "diet": "Omega-3 fatty acids, Vitamin D-rich foods, Garlic, Ginger",
        "meditation": "Pranayama, Relaxation breathing techniques"
    },
    "pneumonia": {
        "symptoms": ["fever", "chills", "cough with phlegm", "shortness of breath"],
        "medication": "Antibiotics, Cough medicine, Pain relievers",
        "diet": "Warm soups, Garlic, Honey, Probiotics",
        "meditation": "Mindfulness meditation, Deep breathing exercises"
    },
    "arthritis": {
        "symptoms": ["joint pain", "swelling", "stiffness", "reduced mobility"],
        "medication": "NSAIDs, Corticosteroids, Disease-modifying drugs",
        "diet": "Turmeric, Omega-3 fatty acids, Green tea, Whole grains",
        "meditation": "Yoga, Tai Chi, Gentle stretching"
    },
    "depression": {
        "symptoms": ["persistent sadness", "loss of interest", "fatigue", "sleep disturbances"],
        "medication": "SSRIs, SNRIs, Therapy, Lifestyle changes",
        "diet": "Omega-3 fatty acids, Dark chocolate, Berries, Nuts",
        "meditation": "Mindfulness meditation, Gratitude journaling, Breathing exercises"
    },
    "anxiety": {
        "symptoms": ["excessive worry", "restlessness", "rapid heartbeat", "trouble sleeping"],
        "medication": "Benzodiazepines, SSRIs, Cognitive-behavioral therapy",
        "diet": "Herbal teas, Magnesium-rich foods, Dark chocolate, Bananas",
        "meditation": "Guided meditation, Deep breathing, Visualization techniques"
    }
}


# Function to Extract Medical Entities using NER
def extract_medical_entities(text):
    entities = ner_pipeline(text)
    medical_terms = [entity["word"] for entity in entities if entity["entity"] in ["SYMPTOM", "DISEASE", "MEDICATION"]]
    return medical_terms

# Function to Find Disease Based on Symptoms
def find_disease(symptoms):
    for disease, info in medical_data.items():
        if any(symptom.lower() in info["symptoms"] for symptom in symptoms):
            return disease, info
    return "Unknown", None

# Function to Generate Dynamic Response
def generate_response(disease, info):
    response = f"**Possible Disease:** {disease.capitalize()}\n\n"
    response += f"**Medication:** {info['medication']}\n\n"
    response += f"**Diet Plan:** {info['diet']}\n\n"
    response += f"**Meditation Suggestion:** {info['meditation']}\n\n"
    response += "**Additional Advice:** It's important to rest and stay hydrated. If symptoms persist, consult a healthcare professional."
    return response

# Streamlit UI
st.title("🩺 REX Medical AI Chatbot")
st.write("Enter symptoms to get possible diseases, medications, and health advice.")

user_input = st.text_input("Enter your symptoms (comma separated):")

if st.button("Get Diagnosis"):
    if user_input.strip() == "":
        st.warning("Please enter some symptoms.")
    else:
        # Extract Medical Entities
        symptoms_list = [s.strip().lower() for s in user_input.split(",")]
        medical_entities = extract_medical_entities(user_input)
        st.write(f"**Extracted Medical Terms:** {', '.join(medical_entities)}")
        
        # Find Disease
        disease, info = find_disease(symptoms_list)
        
        if info:
            # Generate Response
            response = generate_response(disease, info)
            st.markdown(response)
        else:
            st.error("No matching disease found. Please consult a healthcare professional for accurate diagnosis.")