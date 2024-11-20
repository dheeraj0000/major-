import streamlit as st
import base64
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

# Set page configuration
st.set_page_config(
    page_title="English to Hindi Translator",
    page_icon="🌐",
    layout="wide"
)


# Function to set background image
def set_background_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()

        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpeg;base64,{encoded_string}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            .main-container {{
                background-color: rgba(255, 255, 255, 0.8);
                border-radius: 15px;
                padding: 20px;
                margin: 20px;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        st.warning(f"Could not set background image: {e}")


# Custom CSS for styling
def local_css():
    st.markdown("""
    <style>
        .stTextArea {
            background-color: rgba(255, 255, 255, 0.9);
            border-radius: 10px;
            padding: 10px;
        }
        .stButton>button {
            background-color: #2196F3;
            color: white;
            border-radius: 8px;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #1976D2;
            transform: scale(1.05);
        }
        .translation-result {
            background-color: rgba(240, 248, 255, 0.9);
            border-radius: 10px;
            padding: 15px;
            margin-top: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
    </style>
    """, unsafe_allow_html=True)


# Load the tokenizer and model
@st.cache_resource
def load_model():
    try:
        model_directory = r"C:\Users\DRUVA KUMAR\OneDrive\Desktop\MajorProject\tf_model"
        tokenizer = AutoTokenizer.from_pretrained(model_directory)
        model = TFAutoModelForSeq2SeqLM.from_pretrained(model_directory)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


# Main Streamlit app
def main():
    # Apply local CSS
    local_css()

    # Try to set background image (optional)
    try:
        set_background_image(r"C:\Users\DRUVA KUMAR\OneDrive\Desktop\background.jpg")
    except Exception:
        pass

    # Main app content
    st.title("🌐 English to Hindi Translator")
    st.markdown("### Translate English text seamlessly into Hindi")

    # Input text area
    input_text = st.text_area(
        "Enter English text:",
        height=200,
        help="Type or paste the English text you want to translate"
    )

    # Translate button
    if st.button("Translate Now", help="Click to translate the text"):
        if input_text.strip():
            try:
                # Load model
                tokenizer, model = load_model()

                if tokenizer is None or model is None:
                    st.error("Failed to load translation model.")
                    return

                # Tokenize input and generate translation
                tokenized = tokenizer([input_text], return_tensors="np")
                output = model.generate(**tokenized, max_length=128)

                # Decode and display translation
                translated_text = tokenizer.decode(output[0], skip_special_tokens=True)

                # Custom styled result display
                st.markdown(
                    f'<div class="translation-result">'
                    f'<h3 style="color:#1976D2;">🌈 Hindi Translation:</h3>'
                    f'<p style="font-size:18px;">{translated_text}</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            except Exception as e:
                st.error(f"An error occurred during translation: {e}")
        else:
            st.warning("Please enter some text to translate.")

    # Additional information section
    st.markdown("---")
    st.markdown("#### 📝 Translation Tips")
    st.markdown("""
    - Ensure clear and grammatically correct English input
    - Avoid very complex or idiomatic expressions
    - The translation quality depends on the training data
    """)


# Run the app
if __name__ == "__main__":
    main()