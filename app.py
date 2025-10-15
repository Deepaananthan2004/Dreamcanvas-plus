import streamlit as st
from PIL import Image
import os
from utils.caption import get_caption
from utils.emotion import get_dominant_color, infer_emotion_from_color

# Hugging Face imports
from transformers import pipeline
from moviepy.editor import ImageClip, AudioFileClip
import torch
import numpy as np
import cv2
from diffusers import DiffusionPipeline

st.set_page_config(page_title="DreamCanvas+", page_icon="🎨")
st.title("🎨 DreamCanvas+: AI Story from Kids' Drawings")

os.makedirs("outputs", exist_ok=True)

# Hugging Face pipelines
story_generator = pipeline("text2text-generation", model="google/flan-t5-large")
tts_pipeline = pipeline("text-to-speech", model="espnet/kan-bayashi_ljspeech_vits")

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

            # Generate story using Hugging Face
            with st.spinner("🧠 Generating story..."):
                story_input = f"Create a children's story based on this caption: '{caption}' and emotion: '{emotion}'"
                story = story_generator(story_input, max_length=300)[0]['generated_text']
                st.text_area("📖 Story", story, height=150)

            # Generate voice
            with st.spinner("🎤 Generating voice..."):
                tts_output = tts_pipeline(story)
                audio_path = os.path.join("outputs", "story.wav")
                tts_output["wav"].save(audio_path)
                st.audio(audio_path)

            # Generate video from image + audio
            with st.spinner("🎞️ Generating video..."):
                # Using image static clip + audio
                clip = ImageClip(image_path).set_duration(5)
                audio_clip = AudioFileClip(audio_path)
                clip = clip.set_audio(audio_clip).set_duration(audio_clip.duration)
                final_video_path = os.path.join("outputs", "final_video.mp4")
                clip.write_videofile(final_video_path, fps=24)
                st.video(final_video_path)

            # Download buttons
            with open(final_video_path, "rb") as f_vid:
                st.download_button("📥 Download Video", f_vid, file_name="dreamcanvas_video.mp4")
            with open(audio_path, "rb") as f_audio:
                st.download_button("📥 Download Audio", f_audio, file_name="dreamcanvas_audio.wav")

        except Exception as e:
            st.error(f"⚠️ Something went wrong: {e}")
