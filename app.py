# dreamcanvas_plus_app.py

import os
import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
from dotenv import load_dotenv

load_dotenv()  # Load Hugging Face API token from .env
HF_TOKEN = os.getenv("HF_TOKEN")

st.set_page_config(page_title="DreamCanvas+", page_icon="🎨")
st.title("🎨 DreamCanvas+: AI Story from Kids' Drawings")

os.makedirs("outputs", exist_ok=True)


# ---------------------- Hugging Face Helpers ---------------------- #

def hf_text_generation(prompt, model="gpt2"):
    """
    Generate text using Hugging Face Inference API.
    """
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": prompt, "options": {"wait_for_model": True}}
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        try:
            return response.json()[0]["generated_text"]
        except:
            return prompt
    else:
        st.warning(f"⚠️ HF text gen failed: {response.status_code}")
        return prompt


def hf_tts(text, model="tts_transformer", output_path="outputs/audio.wav"):
    """
    Generate TTS using Hugging Face.
    """
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": text}
    response = requests.post(url, headers=headers)
    if response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(response.content)
        return output_path
    else:
        st.warning(f"⚠️ HF TTS failed: {response.status_code}")
        return None


# ---------------------- Image Helpers ---------------------- #

def get_caption(image: Image.Image):
    """
    Simple caption placeholder (replace with HF model if you want real captions)
    """
    return hf_text_generation("Describe this drawing: ")


def get_dominant_color(image_path):
    """
    Simple dominant color detection
    """
    img = Image.open(image_path)
    img = img.convert("RGB")
    colors = img.getcolors(img.size[0]*img.size[1])
    dominant_color = max(colors, key=lambda x: x[0])[1]
    return dominant_color


def infer_emotion_from_color(color):
    """
    Map color to emotion (simple heuristic)
    """
    r, g, b = color
    if r > 200 and g < 100:
        return "excited"
    elif b > 150:
        return "calm"
    elif g > 150:
        return "happy"
    else:
        return "neutral"


def generate_story(caption, emotion):
    """
    Generate story text
    """
    prompt = f"Create a short children's story inspired by this caption: '{caption}' with emotion '{emotion}'"
    return hf_text_generation(prompt)


def auto_generate_description(caption, emotion, story):
    summary = story.strip().split("\n")[0]
    return f"""
✨ Dive into a magical tale born from a child’s imagination!

🎨 Drawing inspired: "{caption}"
🎭 Emotion detected: {emotion.capitalize()}
📖 Story Summary: {summary}

🧒 Voice generated using Hugging Face TTS.
🎬 Video created with DreamCanvas+: GenAI-powered storytelling from kids' art.
""".strip()


def generate_final_video(image_path, audio_path, output_path="outputs/final_video.mp4"):
    """
    Generate a simple video from image + audio
    """
    audio_clip = AudioFileClip(audio_path)
    image_clip = ImageClip(image_path).set_duration(audio_clip.duration)
    image_clip = image_clip.set_audio(audio_clip)
    image_clip.write_videofile(output_path, fps=24)
    return output_path


# ---------------------- Streamlit App ---------------------- #

uploaded = st.file_uploader("Upload your child's drawing", type=["jpg", "png", "jpeg"])

if uploaded:
    image_path = os.path.join("outputs", uploaded.name)
    with open(image_path, "wb") as f:
        f.write(uploaded.read())
    st.image(image_path, caption="Drawing Uploaded", use_column_width=True)

    if st.button("✨ Create Story Video"):
        try:
            with st.spinner("🔍 Captioning drawing..."):
                caption = get_caption(Image.open(image_path))

            color = get_dominant_color(image_path)
            emotion = infer_emotion_from_color(color)

            st.success(f"📝 Caption: {caption}")
            st.success(f"🎭 Emotion: {emotion}")

            with st.spinner("🧠 Generating story..."):
                story = generate_story(caption, emotion)
                st.text_area("📖 Story", story, height=150)

            with st.spinner("🎤 Generating voice..."):
                audio_path = hf_tts(story)
                if audio_path:
                    st.audio(audio_path)
                else:
                    st.warning("⚠️ TTS failed. Video will be generated without audio.")

            if audio_path:
                with st.spinner("🎞️ Generating video..."):
                    final_video = generate_final_video(image_path, audio_path)
                    st.video(final_video)

                with st.spinner("📝 Generating description..."):
                    desc = auto_generate_description(caption, emotion, story)
                    st.text_area("📄 Video Description", desc, height=200)

                # Download buttons
                with open(final_video, "rb") as f_vid:
                    st.download_button("📥 Download Video", f_vid, file_name="dreamcanvas_video.mp4")
                with open(audio_path, "rb") as f_audio:
                    st.download_button("📥 Download Audio", f_audio, file_name="dreamcanvas_audio.wav")

        except Exception as e:
            st.error(f"⚠️ Something went wrong: {e}")
