
import streamlit as st
import requests
from PIL import Image
from style import STYLE, INFO, Guidelines, about
# Configure page
st.set_page_config(
    page_title="Waste Classifier",
    page_icon="♻️",
    layout="centered"
)

# API configuration
API_URL = "http://localhost:8000"
PREDICT_ENDPOINT = f"{API_URL}/predict"

# Custom CSS
st.markdown(STYLE, unsafe_allow_html=True)


def verify_connection():
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def predict_image(uploaded_file):
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        response = requests.post(PREDICT_ENDPOINT, files=files, timeout=30)

        if response.status_code == 200:
            return response.json()["prediction"]
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


def main():
    """Main application function."""

    # Title and subtitle
    st.markdown('<h1 class="main-title">♻️ Waste Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Upload an image to classify waste type</p>', unsafe_allow_html=True)

    # Check API connection
    api_connected = verify_connection()

    if not api_connected:
        st.error("🔌 **API Server Not Connected**")
        st.markdown("""
        Please start the API server first:
        ```bash
        uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
        ```
        """)
        st.stop()
    else:
        st.success("✅ Connected to API server")

    # File upload section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "📁 Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Supported formats: JPG, JPEG, PNG, BMP"
    )

    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        # Create two columns
        col1, col2 = st.columns([1, 1])

        with col1:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

        with col2:
            # Classify button
            if st.button("🔍 Classify Waste", type="primary", use_container_width=True):
                with st.spinner("🤖 Analyzing image..."):
                    prediction = predict_image(uploaded_file)

                if prediction:
                    # Display result
                    st.markdown(f'''
                    <div class="result-container">
                        <p class="result-text">{prediction.title()}</p>
                    </div>
                    ''', unsafe_allow_html=True)

                    # Recycling information
                    recycling_info =INFO

                    info = recycling_info.get(prediction.lower(), {
                        'icon': '❓',
                        'tip': 'Check local guidelines',
                        'bin': 'Check local guidelines'
                    })

                    # Display recycling info
                    st.markdown("### ♻️ Recycling Information")

                    info_col1, info_col2 = st.columns([1, 2])
                    with info_col1:
                        st.markdown(f"## {info['icon']}")
                    with info_col2:
                        st.markdown(f"**Tip:** {info['tip']}")
                        st.markdown(f"**Dispose in:** {info['bin']}")

    # Footer information
    st.markdown("---")

    # Instructions
    with st.expander("📖 How to use this app"):
        st.markdown(Guidelines)

    # About section
    with st.expander("ℹ️ About"):
        st.markdown(about)


if __name__ == "__main__":
    main()