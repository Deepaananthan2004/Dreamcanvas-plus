def auto_generate_description(caption, emotion, story):
    summary = story.strip().split("\n")[0]

    # Build the description
    description = f"""
✨ Dive into a magical tale born from a child’s imagination!

🎨 Drawing inspired: "{caption}"
🎭 Emotion detected: {emotion.capitalize()}
📖 Story Summary: {summary}

🧒 Voice generated using ElevenLabs AI.
🎬 Video created with DreamCanvas+: GenAI-powered storytelling from kids' art.
""".strip()

    return description
