import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import torch
from diffusers import StableDiffusionPipeline
import cv2
from PIL import Image
from serpapi import GoogleSearch

# --- Backend ---

@st.cache_resource  # Cache the model loading
def load_model():
    try:
        model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32).to("cuda" if torch.cuda.is_available() else "cpu")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None  # Return None if model loading fails

def generate_image(prompt, model):
    try:
        image = model(prompt).images[0]
        return image
    except Exception as e:
        st.error(f"Error generating image: {e}")
        return None

def fetch_urls(query):
    try:
        params = {
            "q": query,
            "api_key": "YOUR_SERPAPI_KEY"  # Replace with your SerpAPI key
        }
        search = GoogleSearch(params)
        results = search.get("organic_results", [])
        return [result["link"] for result in results[:5]]
    except Exception as e:
        st.error(f"Error fetching URLs: {e}")
        return []

# --- Frontend ---

class ARTransformer(VideoTransformerBase):
    def __init__(self):
        self.cv2_setUseOptimized = True

    def transform(self, frame):
        cv2.setUseOptimized(self.cv2_setUseOptimized)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.rectangle(frame_rgb, (50, 50), (300, 300), (0, 255,0), 2)  # Example AR effect
        return frame_rgb

st.set_page_config(page_title="Couture AI", layout="wide")

def main():
    # Sidebar menu
    with st.sidebar:
        st.markdown("### ☰ Menu")
        menu_option = st.radio("Navigation", ["Home", "Login", "Your Products"])

    # Top right liked products icon (placeholder)
    st.markdown(
        """
        <style>
            .top-right {
                position: absolute;
                top: 10px;
                right: 20px;
                font-size: 30px;
                cursor: pointer; /* Make it clickable */
            }
        </style>
        <div class='top-right'>fuck</div>  
        """,
        unsafe_allow_html=True,
    )

    if menu_option == "Home":
        # Hero section
        st.markdown(
            """
            <style>
                .hero {
                    background-image: url('https://source.unsplash.com/1600x900/?fashion,model');  /* Or your image URL */
                    background-size: cover;
                    text-align: center;
                    padding: 100px;
                    color: white;
                    font-size: 50px;
                    font-weight: bold;
                }
            </style>
            <div class='hero'>Couture AI: AI-Powered Clothing Design</div>
            """,
            unsafe_allow_html=True,
        )

        # Video section
        st.video("https://youtu.be/cWGxr3fszHI?si=YxP10f3RkL1rp0OE")  # Replace with your video URL

        # User input
        prompt = st.text_input("Enter a prompt for your clothing design:")

        if prompt:
            model = load_model()  # Load the model only if a prompt is entered
            if model:  # Check if model loaded successfully
                with st.spinner("Generating your design..."):
                    image = generate_image(prompt, model)
                if image:  # Check if image was generated successfully
                    st.image(image, caption="Generated Design", use_column_width=True)

                with st.spinner("Fetching relevant links..."):
                    urls = fetch_urls(prompt)

                if urls:
                    st.write("### Relevant Links:")
                    for url in urls:
                        st.markdown(f"- [{url}]({url})")
                else:
                    st.write("No relevant links found.")

                st.write("### 3D Model Generation Coming Soon!")

                st.write("### Try it on with AR!")
                webrtc_streamer(
                    key="ar",
                    transform=ARTransformer,
                    rtc_configuration={
                        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                    }
                )

    elif menu_option == "Login":
        st.subheader("Login Page")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            st.success("Logged in successfully!")  # Placeholder

    elif menu_option == "Your Products":
        st.subheader("Your Previously Designed Products")
        st.write("Coming Soon!")

if __name__ == "__main__":
    main()