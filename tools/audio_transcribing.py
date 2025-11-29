from langchain_core.tools import tool
import speech_recognition as sr
from pydub import AudioSegment
import os

@tool
def transcribe_audio(file_path: str) -> str:
    """
    Transcribe an audio file (.mp3, .wav, .opus, .ogg) into text using Google's Web Speech API.
    
    IMPORTANT: After downloading an audio file, ALWAYS use this tool to transcribe it.
    The transcription will contain instructions on what to do with other data (like CSV files).

    Args:
        file_path (str): Path to the audio file. Just provide the filename (e.g., "demo-audio.opus"),
                        the tool will automatically look in the LLMFiles directory.

    Returns:
        str: The transcribed text from the audio containing instructions.

    Notes:
        - All audio formats are automatically converted to WAV before transcription.
        - Requires internet connection for Google Speech API.
        - Use this immediately after downloading audio files.
    """
    try:
        print(f"[TRANSCRIBE] Starting transcription for: {file_path}")
        # Normalize path - add LLMFiles if not already present
        if not file_path.startswith("LLMFiles"):
            file_path = os.path.join("LLMFiles", file_path)
        
        print(f"[TRANSCRIBE] Full path: {file_path}")
        if not os.path.exists(file_path):
            error_msg = f"Error: File not found at {file_path}"
            print(f"[TRANSCRIBE] {error_msg}")
            return error_msg
        
        final_path = file_path
        
        # Convert audio formats to WAV if needed
        print(f"[TRANSCRIBE] File extension: {file_path.lower()}")
        if file_path.lower().endswith(".mp3"):
            print("[TRANSCRIBE] Converting from MP3 to WAV...")
            sound = AudioSegment.from_mp3(file_path)
            final_path = file_path.replace(".mp3", ".wav")
            sound.export(final_path, format="wav")
        elif file_path.lower().endswith(".opus") or file_path.lower().endswith(".ogg"):
            # OPUS files need to be read without specifying format - pydub auto-detects
            print(f"[TRANSCRIBE] Converting from {file_path.split('.')[-1].upper()} to WAV...")
            sound = AudioSegment.from_file(file_path)  # Auto-detect format
            ext = file_path.split('.')[-1]
            final_path = file_path.replace(f".{ext}", ".wav")
            sound.export(final_path, format="wav")

        # Speech recognition
        print(f"[TRANSCRIBE] Starting Google Speech Recognition on: {final_path}")
        recognizer = sr.Recognizer()
        with sr.AudioFile(final_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)

        print(f"[TRANSCRIBE] ✓ SUCCESS: {text}")
        # If we converted the file, remove temp wav
        if final_path != file_path and os.path.exists(final_path):
            os.remove(final_path)

        return text
    except Exception as e:
        error_msg = f"Error transcribing audio: {str(e)}"
        print(f"[TRANSCRIBE] ✗ FAILED: {error_msg}")
        return error_msg