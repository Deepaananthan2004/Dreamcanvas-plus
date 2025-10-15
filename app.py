import streamlit as st
from PIL import Image
import os
import numpy as np
from io import BytesIO

# Hugging Face
from transformers import pipeline

# Audio
from gtts import gTTS

# Video
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips

# ---------------------- Streamlit Setup ----------------------
st.set_page_config(page_title="DreamCanvas+", page_icon="🎨")
st.title("🎨 DreamCanvas+: AI Story from Kids' Drawings")

os.makedirs("outputs", exist_ok=True)

uploaded = st.file_uploader("Upload your child's drawing", type=["jpg", "png", "jpeg"])

# ---------------------- Helper Functions ----------------------

def get_caption(image):
    """Generate caption using Hugging Face BLIP model"""
    captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    caption = captioner(image)[0]["generated_text"]
    return caption

def get_dominant_color(image_path):
    """Get dominant color from image"""
    image = Image.open(image_path).convert("RGB")
    image = image.resize((50, 50))
    arr = np.array(image)
    arr = arr.reshape(-1, 3)
    counts = {}
    for pixel in arr:
        key = tuple(pixel)
        counts[key] = counts.get(key, 0) + 1
    dominant_color = max(counts, key=counts.get)
    return dominant_color

def infer_emotion_from_color(color):
    """Simple mapping from color to emotion"""
    r, g, b = color
    if r > 150 and g < 100 and b < 100:
        return "Angry"
    elif r < 100 and g > 150 and b < 100:
        return "Happy"
    elif r < 100 and g < 100 and b > 150:
        return "Sad"
    else:
        return "Neutral"

def generate_story(caption, emotion):
    """Text generation using Hugging Face GPT-2"""
    generator = pipeline("text-generation", model="gpt2")
    prompt = f"Write a short, imaginative story based on this caption: '{caption}' with {emotion} emotion."
    story = generator(prompt, max_length=150, do_sample=True)[0]["generated_text"]
    return story

def tts_story(story):
    """Generate audio from story using gTTS (free)"""
    tts = gTTS(story)
    audio_path = "outputs/story.mp3"
    tts.save(audio_path)
    return audio_path

def generate_final_video(image_path, audio_path):
    """Combine image and audio into video using MoviePy"""
    audio_clip = AudioFileClip(audio_path)
    image_clip = ImageClip(image_path).set_duration(audio_clip.duration)
    image_clip = image_clip.set_audio(audio_clip)
    video_path = "outputs/final_video.mp4"
    image_clip.write_videofile(video_path, fps=24)
    return video_path

def auto_generate_description(caption, emotion, story):
    summary = story.strip().split("\n")[0]
    return f"""
✨ Dive into a magical tale born from a child’s imagination!

🎨 Drawing inspired: "{caption}"
🎭 Emotion detected: {emotion.capitalize()}
📖 Story Summary: {summary}

🧒 Voice generated using gTTS.
🎬 Video created with MoviePy.
""".strip()

# ---------------------- Main App Logic ----------------------

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
                audio_path = tts_story(story)
                st.audio(audio_path)

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
                st.download_button("📥 Download Audio", f_audio, file_name="dreamcanvas_audio.mp3")

        except Exception as e:
            st.error(f"⚠️ Something went wrong: {e}")
