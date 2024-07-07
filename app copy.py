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

warnings.filterwarnings("ignore")

with open("/home/agarwal.aditi/mental_health/rf_model.pkl", 'rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)

# Global variable to store recordings with emotion details
recordings_with_emotion = []

def get_recordings_and_predict(start, end):
    '''
    Return predictions for the call recordings between start and end.
    This is efficient for paginations
    '''
    try:
        page_url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Recordings.json"
        recording_response = twilio_api.http_client.request("GET", page_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
        data = json.loads(recording_response.content)
        recording_sids = [recording["sid"] for recording in data.get("recordings", [])]

        if end > len(recording_sids):
            end = len(recording_sids)

        sub_recordings = recording_sids[start:end]
        recordings_with_emotion = []

        for recording_sid in sub_recordings:
            recording_url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Recordings/{recording_sid}.wav"
            auth = (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            response = requests.get(recording_url, auth=auth)

            if response.status_code == 200:
                # Process the recording using your emotion prediction model
                emotion_prediction, pitch_variation, speaking_rate, percent_pause_time, f0_mean, f0_stdev, hnr, localjitter, localshimmer = predict_emotion(response.content)

                # Get additional information from recording data
                recording_data = get_recording_data(recording_sid)

                # Construct a dictionary with recording information, emotion prediction, and additional data
                recording_info = {
                    "RecordingSID": recording_sid,
                    "EmotionPrediction": emotion_prediction,
                    "DateCreated": recording_data.get("date_created", ""),
                    "Duration": recording_data.get("duration", ""),
                    "Price": recording_data.get("price", ""),
                    "PriceUnit": recording_data.get("price_unit", ""),
                    "RecordingURL": recording_url,
                    "PitchVariation": pitch_variation,
                    "SpeakingRate": speaking_rate,
                    "PercentPauseTime": percent_pause_time,
                    "f0_mean": f0_mean,
                    "f0_stdev": f0_stdev,
                    "hnr": hnr,
                    "localjitter": localjitter,
                    "localshimmer": localshimmer
                }

                recordings_with_emotion.append(recording_info)
            else:
                print(f"Error downloading recording {recording_sid}: {response.status_code} - {response.text}")

        return recordings_with_emotion

    except Exception as e:
        print(f"Error: {str(e)}")
        return []

def calculate_pitch_variation(audio_file_path):
    # Load audio file
    samplerate, samples = read_audio(audio_file_path)
    samples = samples.astype(np.float32)
    
    # Initialize pitch object
    pitch_o = aubio.pitch("yin", samplerate=samplerate)
    
    # Array to store F0 values
    f0_values = []
    
    # Calculate F0 for each frame
    hop_size = 512
    total_frames = len(samples) // hop_size
    for i in range(total_frames):
        samples_frame = samples[i * hop_size:(i + 1) * hop_size]
        pitch = pitch_o(samples_frame)[0]
        if pitch != 0:
            f0_values.append(pitch)
    
    # Calculate average fundamental frequency
    average_f0 = np.mean(f0_values)
    
    # Calculate standard deviation of fundamental frequency
    std_dev_f0 = np.std(f0_values)
    
    # Calculate coefficient of variation (CV) of fundamental frequency
    cv_f0 = std_dev_f0 / average_f0
    
    return cv_f0

def read_audio(audio_file_path):
    from scipy.io import wavfile
    # Read audio file
    samplerate, samples = wavfile.read(audio_file_path)
    # Convert to mono if stereo
    if len(samples.shape) > 1:
        samples = samples.mean(axis=1)
    return samplerate, samples

def calculate_speaking_rate(audio_file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        return None
    except sr.RequestError as e:
        return None
    words = text.split()
    word_count = len(words)
    audio_duration = len(audio_data.get_raw_data()) / (audio_data.sample_width * audio_data.sample_rate)
    speaking_rate = (word_count / audio_duration) * 60
    return speaking_rate

def measurePitch(voiceID, f0min, f0max, unit):
    sound = parselmouth.Sound(voiceID) # read the sound
    duration = call(sound, "Get total duration") # duration
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max) #create a praat pitch object
    meanF0 = call(pitch, "Get mean", 0, 0, unit) # get mean pitch
    stdevF0 = call(pitch, "Get standard deviation", 0 ,0, unit) # get standard deviation
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localShimmer =  call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    return meanF0, stdevF0, hnr, localJitter, localShimmer

def calculate_percent_pause_time(audio_file_path):
    audio = AudioSegment.from_file(audio_file_path)
    silence_threshold = -50
    min_silence_duration = 250
    silent_chunks = split_on_silence(audio, min_silence_len=min_silence_duration, silence_thresh=silence_threshold)
    silent_chunks_filtered = [chunk for chunk in silent_chunks if len(chunk) >= min_silence_duration]
    total_silent_duration = sum(len(chunk) for chunk in silent_chunks_filtered)
    total_duration = len(audio)
    percent_pause_time_score = ((total_duration - total_silent_duration) / total_duration) * 100
    return percent_pause_time_score

def predict_emotion(audio_content):
    # Preprocess audio to extract features
    audio = AudioSegment.from_file(io.BytesIO(audio_content))
    audio_file_path = "temp_audio.wav"  # Temporary file path for audio
    audio.export(audio_file_path, format="wav")

    pitch_variation = calculate_pitch_variation(audio_file_path)
    speaking_rate = calculate_speaking_rate(audio_file_path)
    percent_pause_time = calculate_percent_pause_time(audio_file_path)
    f0_mean, f0_stdev, hnr, localjitter, localshimmer = measurePitch(audio_file_path, 75, 500, "Hertz")

    # Use the extracted features to make emotion prediction
    # You can replace this part with your actual emotion prediction model logic
    # For demonstration purposes, we're just using a placeholder
    prediction = model.predict([[pitch_variation, speaking_rate, percent_pause_time]])
    # Map the predicted label to an emotion
    return prediction[0], pitch_variation, speaking_rate, percent_pause_time, f0_mean, f0_stdev, hnr, localjitter, localshimmer

def get_recording_data(recording_sid):
    # Retrieve recording data using Twilio API
    recording_url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Recordings/{recording_sid}.json"
    auth = (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    response = requests.get(recording_url, auth=auth)

    if response.status_code == 200:
        recording_data = json.loads(response.content)
        return recording_data
    else:
        print(f"Error retrieving recording data for {recording_sid}: {response.status_code} - {response.text}")
        return {}

@app.route("/")
def index():
    global recordings_with_emotion
    # Fetch recordings with emotion details
    recordings_with_emotion = get_recordings_and_predict(0, 2)
    return render_template('index.html', recordings_with_emotion=recordings_with_emotion)

@app.route("/detailed_analysis", methods=["GET", "POST"])
def detailed_analysis():
    if request.method == "POST":
        selected_recording_sid = request.form.get("recording_sid")
        selected_recording = next((recording for recording in recordings_with_emotion if recording["RecordingSID"] == selected_recording_sid), None)
        if selected_recording:
            return render_template('detailed_analysis.html', selected_recording=selected_recording, recordings_with_emotion=recordings_with_emotion)
    
    return render_template('detailed_analysis.html', selected_recording=None, recordings_with_emotion=recordings_with_emotion)

if __name__ == "__main__":
    app.run(debug=True)
