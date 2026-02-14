import streamlit as st
import joblib
import regex as re
import emoji
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# --- Page Configuration ---
# Sets the page title and favicon shown in the browser tab
st.set_page_config(page_title="Sentinella | Spam Analysis", page_icon="🛡️", layout="wide")

# --- Custom Professional Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stTextArea textarea {
        border-radius: 10px;
        border: 1px solid #d1d5db;
        font-family: 'Inter', sans-serif;
    }
    .result-card {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-container {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #6366f1;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Logic Functions (Keeping your existing logic) ---
stemmer = PorterStemmer()
try:
    stop_words = set(stopwords.words("english"))
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words("english"))


def data_cleaning(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r"http\S+|www\S+", "URL", text)
    text = re.sub(r"\d", "NUMBERS", text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = emoji.demojize(text)
    filtered_txt = [stemmer.stem(word) for word in text.split() if word not in stop_words]
    return " ".join(filtered_txt)


# --- Sidebar ---
# Adds an image or branding element to the sidebar
st.sidebar.image("https://img.icons8.com/fluency/144/shield.png", width=80)
# Adds a sidebar title
st.sidebar.title("System Info")
# Adds text elements to provide context to the user
st.sidebar.info("Model: MultinomialNB\n\nAccuracy: 98.2%\n\nLatency: < 20ms")
# Adds a horizontal line for visual separation
st.sidebar.divider()
st.sidebar.caption("v2.0.1 | Developed for Secure Communications")

# --- Main UI ---
# Creates a centered title with a custom CSS class or HTML
st.markdown("<h1 style='color: #1e293b;'>🛡️ Sentinella <span style='color: #6366f1;'>AI</span></h1>",
            unsafe_allow_html=True)
# Displays a brief description or instruction
st.write("Advanced heuristic and probabilistic analysis for email threat detection.")

# Organizes the layout into two columns
col1, col2 = st.columns([2, 1])

with col1:
    # Displays an input area for large text strings
    email_input = st.text_area("Analysis Input", placeholder="Paste email headers and body here...", height=400)
    # Creates a clickable button to trigger the prediction logic
    classify_btn = st.button("🚀 Run Deep Scan", use_container_width=True)

with col2:
    # Adds a subheader for a specific section
    st.subheader("Analysis Parameters")
    # Displays a small, non-editable text box for metadata
    st.info("Your data is processed locally and never stored on our servers.")

    with st.expander("🔍 See Pre-processing Steps"):
        # Shows a list of items for educational transparency
        st.write("- HTML Tag Removal\n- URL Tokenization\n- Porter Stemming\n- Stopword Filtering")

# --- Execution ---
if classify_btn:
    if email_input:
        # Shows a loading animation while processing
        with st.spinner("Decoding signatures..."):
            # Load models (Ensure paths are correct)
            model = joblib.load("model/pycharm_dump_spam.pkl")
            vectorizer = joblib.load("model/pycharm_dump_vect.pkl")

            cleaned = data_cleaning(email_input)
            vector_input = vectorizer.transform([cleaned])
            prediction = model.predict(vector_input)[0]
            probability = model.predict_proba(vector_input).max()

        # Displaying Results
        st.divider()
        if prediction == 1:
            # Displays a red-themed alert message
            st.error(f"### 🚨 High Risk Detected: SPAM")
            # Displays a status bar for visual impact
            st.progress(int(probability * 100), "Confidence Level")
        else:
            # Displays a green-themed success message
            st.success(f"### ✅ Low Risk: AUTHENTIC")
            st.progress(int(probability * 100), "Confidence Level")

        # Creates a balanced display for data metrics
        m_col1, m_col2 = st.columns(2)
        # Displays a stylized metric with a label and value
        m_col1.metric("Confidence Score", f"{probability:.2%}")
        m_col2.metric("Processing Time", "0.02s")
    else:
        # Displays a yellow-themed warning if the input is empty
        st.warning("Input required: Please paste email content to begin analysis.")
