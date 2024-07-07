import streamlit as st
import numpy as np
import wave
import matplotlib.pyplot as plt
import pandas as pd
import os
import warnings
from flask import Flask, render_template, make_response, request
from twilio.rest import Client
from dotenv import load_dotenv
import requests
import json
import numpy as np
import io
import pickle
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from pydub import AudioSegment
from pydub.silence import split_on_silence
import speech_recognition as sr
import aubio
import parselmouth
from parselmouth.praat import call

load_dotenv()  # load environment variables

TWILIO_ACCOUNT_SID = os.environ["TWILIO_ACCOUNT_SID"]
TWILIO_AUTH_TOKEN = os.environ["TWILIO_AUTH_TOKEN"]
twilio_api = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

with open("D:/Aayush/Aayush/Mental Health Project/OneDrive_2024-05-14/mental health/rf_model.pkl", 'rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)
page_url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Recordings.json"
# print(page_url)
recording_response = twilio_api.http_client.request("GET", page_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
# print(recording_response.content)
data = json.loads(recording_response.content)
recording_sids = [recording["sid"] for recording in data.get("recordings", [])]
end = 1
start = 0
if end > len(recording_sids):
    end = len(recording_sids)

sub_recordings = recording_sids[start:end]
# print(sub_recordings)
recordings_with_emotion = []
import wave
def create_wav_file(audio_content):
    a = io.BytesIO(audio_content)
    audio_data = a.getvalue()
    nchannels = 1  # number of audio channels (1 for mono, 2 for stereo)
    sampwidth = 2  # sample width in bytes (1 for 8-bit, 2 for 16-bit, etc.)
    framerate = 8000  # frames per second
    nframes = len(audio_data) // (sampwidth * nchannels)  # number of frames

    # # Create a new WAV file
    with wave.open("temp_audio_1.wav", "wb") as audio:
        audio.setnchannels(nchannels)
        audio.setsampwidth(sampwidth)
        audio.setframerate(framerate)
        audio.setnframes(nframes)
        audio.writeframes(audio_data)

    print("WAV file written successfully.")
    # print(type(audio))
    audio_file_path = "temp_audio_1.wav"  # Temporary file path for audio
    return audio_file_path

for recording_sid in sub_recordings:
    recording_url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Recordings/{recording_sid}.wav"
    print(recording_url)
    auth = (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    response = requests.get(recording_url, auth=auth)
    # print(response.status_code)
    # print(response.content)
    if response.status_code == 200:
        # Process the recording using your emotion prediction model
        audio_file_path = create_wav_file(response.content)


# Display the header
text = "Voice Acoustic Features"
st.markdown(
    f"""
    <h2 style="font-size:50px;">{text}</h2>
    """,
    unsafe_allow_html=True
)

# Function to plot Amplitude vs Time
def plot_amplitude_vs_time(audio_file_path):
    st.write("Amplitude vs Time")
    # Open the WAV file
    normal_wav_file = wave.open(audio_file_path, 'r')

    # Get the number of frames and the sample rate
    num_frames = normal_wav_file.getnframes()
    sample_rate = normal_wav_file.getframerate()
    num_channels = normal_wav_file.getnchannels()

    # Read the frames and convert them to a numpy array
    frames = normal_wav_file.readframes(num_frames)
    audio_data = np.frombuffer(frames, dtype=np.int16)

    # If the audio is stereo, reshape the array
    if num_channels == 2:
        audio_data = audio_data.reshape(-1, 2)

    # Use only the first channel if stereo
    if num_channels == 2:
        audio_data = audio_data[:, 0]

    # Calculate the time axis
    time = np.arange(0, len(audio_data)) / sample_rate

    # Plot the amplitude vs time graph
    plt.figure(figsize=(10, 4))
    plt.plot(time, audio_data)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Amplitude vs Time')
    st.pyplot(plt)

    # Close the WAV file
    normal_wav_file.close()

# Function to calculate and plot Loudness vs Time
def calculate_loudness(audio_file_path, window_size=1024, overlap=512):
    st.write("Loudness vs Time")
    # Open the WAV file
    normal_wav_file = wave.open(audio_file_path, 'r')

    # Get the number of frames and the sample rate
    num_frames = normal_wav_file.getnframes()
    sample_rate = normal_wav_file.getframerate()
    num_channels = normal_wav_file.getnchannels()

    # Read the frames and convert them to a numpy array
    frames = normal_wav_file.readframes(num_frames)
    audio_data = np.frombuffer(frames, dtype=np.int16)

    # If the audio is stereo, reshape the array
    if num_channels == 2:
        audio_data = audio_data.reshape(-1, 2)
        # Use only the first channel if stereo
        audio_data = audio_data[:, 0]

    # Close the WAV file
    normal_wav_file.close()

    # Initialize lists to store time and loudness values
    times = []
    loudness_values = []

    # Calculate the loudness over time using RMS
    step_size = window_size - overlap
    for start in range(0, len(audio_data) - window_size + 1, step_size):
        window = audio_data[start:start + window_size]
        rms = np.sqrt(np.mean(window.astype(np.float32) ** 2))
        loudness = 20 * np.log10(rms / 32768.0)
        times.append(start / sample_rate)
        loudness_values.append(loudness)

    # Plot the loudness vs time graph
    plt.figure(figsize=(10, 4))
    plt.plot(times, loudness_values)
    plt.xlabel('Time (s)')
    plt.ylabel('Loudness (dB)')
    plt.title('Loudness vs Time')
    st.pyplot(plt)

def create_hist(file_path, feature):
    st.write(f"{feature} Histogram")
    data = pd.read_csv(file_path)

    # Filter the data
    data_less_50 = data[data['SDS Score'] < 50]
    data_more_equal_50 = data[data['SDS Score'] >= 50]

    # Plot histograms
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Histogram for sds < 50
    axs[0].hist(data_less_50[feature], bins=20, color='blue', edgecolor='black', weights=[100 / len(data_less_50)] * len(data_less_50))
    axs[0].set_title(f"Percentage Histogram of {feature} (sds < 50)")
    axs[0].set_xlabel(f"{feature}")
    axs[0].set_ylabel('Percentage')

    # Histogram for sds >= 50 (normalized to show percentage)
    axs[1].hist(data_more_equal_50[feature], bins=20, color='green', edgecolor='black', weights=[100 / len(data_more_equal_50)] * len(data_more_equal_50))
    axs[1].set_title('Percentage Histogram of f0 (sds >= 50)')
    axs[1].set_xlabel(f"{feature}")

    # Show the plot
    plt.tight_layout()
    st.pyplot(plt)

# Example usage
plot_amplitude_vs_time(audio_file_path)
calculate_loudness(audio_file_path)
create_hist("new_csv.csv", "F0 Mean")


