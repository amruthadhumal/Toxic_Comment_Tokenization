import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# ---------------------------------------
# Load Saved Model and Tokenizer
# ---------------------------------------
#MODEL_PATH = "Saved_model"
MODEL_PATH = r"\Users\dhuma\OneDrive\Desktop\GUVI\Project_5\Current\Saved_model"

@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Toxicity Labels
LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

# ---------------------------------------
# Prediction Function
# ---------------------------------------
def predict_comment(text):
    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    probabilities = torch.sigmoid(outputs.logits).cpu().numpy()[0]

    results = {}
    for label, prob in zip(LABELS, probabilities):
        results[label] = float(prob)

    return results

# ---------------------------------------
# Streamlit UI
# ---------------------------------------
st.set_page_config(
    page_title="Toxic Comment Detection",
    page_icon="💬",
    layout="centered"
)

st.title("💬 Toxic Comment Detection")
st.write(
    "Enter a comment below and check whether it contains toxic content."
)

user_text = st.text_area(
    "Enter Comment",
    height=150,
    placeholder="Type your comment here..."
)

if st.button("Predict"):

    if not user_text.strip():
        st.warning("Please enter a comment.")
    else:
        predictions = predict_comment(user_text)

        st.subheader("Prediction Results")

        toxic_detected = False

        for label, score in predictions.items():
            st.progress(float(score))
            st.write(f"**{label}** : {score:.4f}")

            if score > 0.5:
                toxic_detected = True

        st.markdown("---")

        if toxic_detected:
            st.error("⚠ Toxic Comment Detected")
        else:
            st.success("✅ Non-Toxic Comment")

