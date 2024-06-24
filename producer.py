import gradio as gr
import assemblyai as aai
from anthropic import Anthropic
import os
from pydub import AudioSegment
import tempfile

# Initialize API clients
aai.settings.api_key = os.getenv("AAI_KEY")
anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
model_name = "claude-3-5-sonnet-20240620"

# Default prompts from the initial Python file
DEFAULT_TITLE_PROMPT = """
Suggest a title for the podcast episode
Dwarkesh is the host, not the guest.
The format should be:
Guest Name - Title

Make it sexy! Enticing! No boilerplate!!

Don't output anything but the suggested podcast title.

Here are some examples of previous podcast episode titles:

Patrick Collison (Stripe CEO) - Craft, Beauty, & The Future of Payments
Tyler Cowen - Hayek, Keynes, & Smith on AI, Animal Spirits, Anarchy, & Growth
Jung Chang - Living through Cultural Revolution and the Crimes of Mao
Andrew Roberts - SV's Napoleon Cult, Why Hitler Lost WW2, Churchill as Applied Historian
Dominic Cummings - COVID, Brexit, & Fixing Western Governance
Paul Christiano - Preventing an AI Takeover
Shane Legg (DeepMind Founder) - 2028 AGI, New Architectures, Aligning Superhuman Models
Grant Sanderson (3Blue1Brown) - Past, Present, & Future of Mathematics
Sarah C. M. Paine - WW2, Taiwan, Ukraine, & Maritime vs Continental Powers
Dario Amodei (Anthropic CEO) - Scaling, Alignment, & AI Progress

Come up with a title for the following transcript using guidance above.

Come up 5-10 titles, each on a new line, so I can select the best one.
"""

DEFAULT_CLIP_PROMPT = """
TODO
"""


def transcribe_audio(audio_file):
    # Handle both file objects and file paths
    if isinstance(audio_file, str):
        audio_path = audio_file
    else:
        audio_path = audio_file.name

    # Convert audio to MP3 if it's not already
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
        audio = AudioSegment.from_file(audio_path)
        audio.export(temp_file.name, format="mp3")

        # Transcribe the audio file
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(
            temp_file.name, config=aai.TranscriptionConfig(speaker_labels=True)
        )

    # Format the transcript
    formatted_transcript = ""
    for utterance in transcript.utterances:
        formatted_transcript += f"{utterance.speaker} {format_timestamp(utterance.start)}\n{utterance.text}\n\n"

    return formatted_transcript


def format_timestamp(milliseconds):
    seconds = int(milliseconds / 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def generate_titles(transcript, prompt):
    message = anthropic.messages.create(
        model=model_name,
        max_tokens=1024,
        messages=[
            {"role": "user", "content": f"{prompt}\n\nTranscript:\n{transcript}"},
        ],
    )
    return message.content[0].text


def generate_clips(transcript, prompt):
    message = anthropic.messages.create(
        model=model_name,
        max_tokens=1024,
        messages=[
            {"role": "user", "content": f"{prompt}\n\nTranscript:\n{transcript}"},
        ],
    )
    return message.content[0].text


def process_transcript(audio_file):
    transcript = transcribe_audio(audio_file)

    # Save transcript to a temporary file
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".txt"
    ) as temp_file:
        temp_file.write(transcript)
        temp_file_path = temp_file.name

    return transcript, temp_file_path


def process_title_and_clips(transcript, title_prompt, clip_prompt):
    titles = generate_titles(transcript, title_prompt)
    clips = generate_clips(transcript, clip_prompt)
    return titles, clips


# Define the Gradio interface
with gr.Blocks() as app:
    gr.Markdown("# Podcast Helper")

    # with gr.Row():
    audio_input = gr.Audio(type="filepath", label="Upload MP3")
    transcribe_button = gr.Button("Generate Transcript")

    with gr.Row():
        transcript_output = gr.Textbox(label="Transcript", lines=10)
        transcript_file = gr.File(label="Download Transcript")

    with gr.Row():
        title_prompt = gr.Textbox(
            label="Title Generation Prompt", lines=3, value=DEFAULT_TITLE_PROMPT
        )
        clip_prompt = gr.Textbox(
            label="Clip Search Prompt", lines=3, value=DEFAULT_CLIP_PROMPT
        )

    generate_button = gr.Button("Generate Title and Clips")

    with gr.Row():
        titles_output = gr.Textbox(label="Generated Titles", lines=10)
        clips_output = gr.Textbox(label="Generated Clips", lines=10)

    transcribe_button.click(
        process_transcript,
        inputs=[audio_input],
        outputs=[transcript_output, transcript_file],
    )

    generate_button.click(
        process_title_and_clips,
        inputs=[transcript_output, title_prompt, clip_prompt],
        outputs=[titles_output, clips_output],
    )

# Launch the app
if __name__ == "__main__":
    app.launch()
