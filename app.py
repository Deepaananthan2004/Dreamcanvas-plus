import streamlit as st
from PIL import Image
import os
import requests
from io import BytesIO
from dotenv import load_dotenv
import colorsys

# Load .env
load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

st.set_page_config(page_title="DreamCanvas+", page_icon="🎨")
st.title("🎨 DreamCanvas+: AI Story from Kids' Drawings")

os.makedirs("outputs", exist_ok=True)

uploaded = st.file_uploader("Upload your child's drawing", type=["jpg", "png", "jpeg"])

def get_caption(image):
    """Generate image caption using Hugging Face model"""
    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes = image_bytes.getvalue()
    url = "https://api-inference.huggingface.co/models/nlpconnect/vit-gpt2-image-captioning"
    response = requests.post(url, headers=HEADERS, files={"image": image_bytes})
    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        return "A child’s beautiful drawing."

def get_dominant_color(image_path):
    """Get dominant color from image for emotion inference"""
    image = Image.open(image_path).convert("RGB")
    image = image.resize((50,50))
    pixels = list(image.getdata())
    r = sum([p[0] for p in pixels])/len(pixels)
    g = sum([p[1] for p in pixels])/len(pixels)
    b = sum([p[2] for p in pixels])/len(pixels)
    return (r,g,b)

def infer_emotion_from_color(rgb):
    """Map dominant color to simple emotions"""
    r,g,b = rgb
    h,l,s = colorsys.rgb_to_hls(r/255,g/255,b/255)
    if l < 0.4:
        return "Sad"
    elif s < 0.2:
        return "Calm"
    elif h < 0.1 or h > 0.9:
        return "Angry"
    elif 0.1 < h < 0.4:
        return "Happy"
    else:
        return "Excited"

def generate_story(caption, emotion):
    """Generate story using Hugging Face text-generation"""
    prompt = f"Write a short magical story inspired by this caption: '{caption}' with emotion: {emotion}"
    url = "https://api-inference.huggingface.co/models/gpt2"
    response = requests.post(url, headers=HEADERS, json={"inputs": prompt})
    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        return "Once upon a time, a child drew a magical adventure..."

def elevenlabs_tts(text):
    """Generate speech using Hugging Face TTS model"""
    url = "https://api-inference.huggingface.co/models/facebook/fastspeech2-en-ljspeech"
    response = requests.post(url, headers=HEADERS, json={"inputs": text})
    if response.status_code == 200:
        audio_bytes = BytesIO(response.content)
        audio_path = os.path.join("outputs", "story_audio.wav")
        with open(audio_path, "wb") as f:
            f.write(audio_bytes.getbuffer())
        return audio_path
    else:
        st.warning("TTS generation failed.")
        return None

def generate_final_video(image_path, audio_path):
    """Combine image and audio into video using Hugging Face video model"""
    url = "https://api-inference.huggingface.co/models/facebook/animated-video"
    with open(image_path, "rb") as img_file, open(audio_path, "rb") as audio_file:
        response = requests.post(url, headers=HEADERS, files={"image": img_file, "audio": audio_file})
    if response.status_code == 200:
        video_path = os.path.join("outputs", "story_video.mp4")
        with open(video_path, "wb") as f:
            f.write(response.content)
        return video_path
    else:
        st.warning("Video generation failed.")
        return None

def auto_generate_description(caption, emotion, story):
    summary = story.strip().split("\n")[0]
    return f"""
✨ Dive into a magical tale born from a child’s imagination!

🎨 Drawing inspired: "{caption}"
🎭 Emotion detected: {emotion.capitalize()}
📖 Story Summary: {summary}

🧒 Voice generated using Hugging Face TTS.
🎬 Video created using Hugging Face Video model.
""".strip()

if uploaded:
    image_path = os.path.join("outputs", uploaded.name)
    with open(image_path, "wb") as f:
        f.write(uploaded.read())
    st.image(image_path, caption="Drawing Uploaded", use_column_width=True)

    if st.button("✨ Create Story Video"):
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
            audio_path = elevenlabs_tts(story)
            if audio_path:
                st.audio(audio_path)

        with st.spinner("🎞️ Generating video..."):
            final_video = generate_final_video(image_path, audio_path)
            if final_video:
                st.video(final_video)

        with st.spinner("📝 Generating description..."):
            desc = auto_generate_description(caption, emotion, story)
            st.text_area("📄 Video Description", desc, height=200)

        # Download buttons
        if final_video:
            with open(final_video, "rb") as f_vid:
                st.download_button("📥 Download Video", f_vid, file_name="dreamcanvas_video.mp4")
        if audio_path:
            with open(audio_path, "rb") as f_audio:
                st.download_button("📥 Download Audio", f_audio, file_name="dreamcanvas_audio.wav")
