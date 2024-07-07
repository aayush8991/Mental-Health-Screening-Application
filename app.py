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
import wave
from parselmouth.praat import call

load_dotenv()  # load environment variables

TWILIO_ACCOUNT_SID = os.environ["TWILIO_ACCOUNT_SID"]
TWILIO_AUTH_TOKEN = os.environ["TWILIO_AUTH_TOKEN"]
twilio_api = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

warnings.filterwarnings("ignore")

# D:\Aayush\Aayush\Mental Health Project\OneDrive_2024-05-14\mental health\rf_model.pkl

with open("D:/Aayush/Aayush/Mental Health Project/OneDrive_2024-05-14/mental health/rf_model.pkl", 'rb') as model_file:
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
        print(page_url)
        recording_response = twilio_api.http_client.request("GET", page_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
        # print(recording_response.content)
        data = json.loads(recording_response.content)
        # print(data)
        recording_sids = [recording["sid"] for recording in data.get("recordings", [])]
        # print(len(recording_sids))
        # print(recording_sids)

        if end > len(recording_sids):
            end = len(recording_sids)

        sub_recordings = recording_sids[start:end]
        # print(sub_recordings)
        recordings_with_emotion = []
        # print(len(sub_recordings))
        # print(sub_recordings)
        for recording_sid in sub_recordings:
            recording_url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Recordings/{recording_sid}.wav"
            # print(recording_url)
            auth = (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            response = requests.get(recording_url, auth=auth)
            # print(response.status_code)
            # print(response.content)
            if response.status_code == 200:
                # Process the recording using your emotion prediction model
                emotion_prediction, pitch_variation, speaking_rate, percent_pause_time, f0_mean, f0_stdev, hnr, localjitter, localshimmer = predict_emotion(response.content)

                # Get additional information from recording data
                recording_data = get_recording_data(recording_sid)
                call_sid = recording_data.get("call_sid", "")
                call = twilio_api.calls(call_sid).fetch()
                ToPhoneNumber = (call.to)
                # Construct a dictionary with recording information, emotion prediction, and additional data
                recording_info = {
                    "RecordingSID": recording_sid,
                    "DateCreated": recording_data.get("date_created", ""),
                    "Duration": recording_data.get("duration", ""),
                    "Price": recording_data.get("price", ""),
                    "PriceUnit": recording_data.get("price_unit", ""),
                    "To": ToPhoneNumber,
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

                # print(call_sid)
                recordings_with_emotion.append(recording_info)
                # recordings_with_emotion.append(call)
            else:
                print(f"Error downloading recording {recording_sid}: {response.status_code} - {response.text}")

        return recordings_with_emotion

    except Exception as e:
        print(f"Error: {str(e)}")
        return []

# def calculate_pitch_variation(audio_file_path):
#     # Load audio file
#     samplerate, samples = read_audio(audio_file_path)
#     samples = samples.astype(np.float32)
    
#     # Initialize pitch object
#     pitch_o = aubio.pitch("yin", samplerate=samplerate)
    
#     # Array to store F0 values
#     f0_values = []
    
#     # Calculate F0 for each frame
#     hop_size = 512
#     total_frames = len(samples) // hop_size
#     for i in range(total_frames):
#         samples_frame = samples[i * hop_size:(i + 1) * hop_size]
#         pitch = pitch_o(samples_frame)[0]
#         if pitch != 0:
#             f0_values.append(pitch)
    
#     # Calculate average fundamental frequency
#     average_f0 = np.mean(f0_values)
    
#     # Calculate standard deviation of fundamental frequency
#     std_dev_f0 = np.std(f0_values)
    
#     # Calculate coefficient of variation (CV) of fundamental frequency
#     cv_f0 = std_dev_f0 / average_f0
    
#     return cv_f0

def calculate_pitch_variation(audio_file_path):
    # Load audio file
    samplerate, samples = read_audio(audio_file_path)
    # print("samplerate=", samplerate)
    
    # Normalize samples to [-1, 1]
    if samples.ndim > 1:
        samples = np.mean(samples, axis=1)
    
    # Normalize samples to [-1, 1]
    samples = samples.astype(np.float32)
    samples = samples / np.max(np.abs(samples))
    # Initialize pitch object with appropriate parameters
    win_s = 1024  # window size
    hop_s = 512   # hop size
    
    pitch_o = aubio.pitch("yin", win_s, hop_s, samplerate)
    pitch_o.set_unit("Hz")
    pitch_o.set_silence(-40)  # silence threshold
    
    # Array to store F0 values
    f0_values = []
    
    # Calculate F0 for each frame
    total_frames = len(samples) // hop_s
    for i in range(total_frames):
        samples_frame = samples[i * hop_s:(i + 1) * hop_s]
        pitch = pitch_o(samples_frame)[0]
        if pitch > 0 and 50 < pitch < 500:  # Filter for human speech pitch range
            f0_values.append(pitch)
    
    # Calculate average fundamental frequency
    if f0_values:  # Ensure there are valid pitch values
        average_f0 = np.mean(f0_values)
        # Calculate standard deviation of fundamental frequency
        std_dev_f0 = np.std(f0_values)
        # Calculate coefficient of variation (CV) of fundamental frequency
        cv_f0 = std_dev_f0 / average_f0
        return cv_f0
    else:
        return 0  # No valid pitch values found

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
        audio_duration = source.DURATION
    try:
        text = recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        return None
    except sr.RequestError as e:
        return None
    words = text.split()
    word_count = len(words)
    # audio_duration = len(audio_data.get_raw_data()) / (audio_data.sample_width * audio_data.sample_rate)
    audio_duration = len(audio_data.get_raw_data()) / (2*8000)
    # print(word_count, audio_duration)
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
    min_silence_duration = 1250
    silent_chunks = split_on_silence(audio, min_silence_len=min_silence_duration, silence_thresh=silence_threshold)
    silent_chunks_filtered = [chunk for chunk in silent_chunks if len(chunk) >= min_silence_duration]
    total_silent_duration = sum(len(chunk) for chunk in silent_chunks_filtered)
    total_duration = len(audio)
    percent_pause_time_score = ((total_duration - total_silent_duration) / total_duration) * 100
    return percent_pause_time_score

def predict_emotion(audio_content):
    # Preprocess audio to extract features
    # print(audio_content)
    # print("Hello from predict_emotion")
    print()
    a = io.BytesIO(audio_content)
    # print(a)
    audio_data = a.getvalue()
    nchannels = 1  # number of audio channels (1 for mono, 2 for stereo)
    sampwidth = 2  # sample width in bytes (1 for 8-bit, 2 for 16-bit, etc.)
    framerate = 8000  # frames per second
    nframes = len(audio_data) // (sampwidth * nchannels)  # total number of frames

# Calculate the number of frames to skip (5 seconds worth of frames)
    skip_duration = 4 # seconds
    frames_to_skip = framerate * skip_duration

    # Ensure we don't try to skip more frames than we have
    if frames_to_skip > nframes:
        frames_to_skip = nframes

    # Calculate the byte offset to start reading audio data after skipping
    byte_offset = frames_to_skip * sampwidth * nchannels
    trimmed_audio_data = audio_data[byte_offset:]

    # Calculate the new number of frames after trimming
    new_nframes = len(trimmed_audio_data) // (sampwidth * nchannels)
    # Create a new WAV file
    with wave.open("temp_audio.wav", "wb") as audio:
        audio.setnchannels(nchannels)
        audio.setsampwidth(sampwidth)
        audio.setframerate(framerate)
        audio.setnframes(new_nframes)
        audio.writeframes(trimmed_audio_data)
    audio_file_path = "temp_audio.wav"

    pitch_variation = calculate_pitch_variation(audio_file_path)
    speaking_rate = calculate_speaking_rate(audio_file_path)
    percent_pause_time = calculate_percent_pause_time(audio_file_path)
    f0_mean, f0_stdev, hnr, localjitter, localshimmer = measurePitch(audio_file_path, 75, 500, "Hertz")
    # print(f0_mean, f0_stdev, hnr, localjitter, localshimmer)
    # Use the extracted features to make emotion prediction
    # You can replace this part with your actual emotion prediction model logic
    # For demonstration purposes, we're just using a placeholder
    # print([[pitch_variation, speaking_rate, percent_pause_time]])
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

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

# D:\Aayush\Aayush\Mental Health Project\OneDrive_2024-05-14\mental health\audio_features_EATD.csv
df = pd.read_csv("D:/Aayush/Aayush/Mental Health Project/OneDrive_2024-05-14/mental health/new_csv.csv")
df['SDS Category'] = pd.cut(df['SDS Score'], bins=[0, 50, float('inf')], labels=['Normal', 'Needs Attention'])

def plot_master(value,feature_name, recording_index,ind):
    features_to_plot = ['Pitch Variation', 'Speaking Rate', 'Percent Pause Time', 
                        'F0 Mean', 'F0 Standard Deviation', 'HNR', 
                        'Local Jitter', 'Local Shimmer']
    feature = features_to_plot[ind]
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='PHQ8', y=feature, data=df)
    plt.title(f'{feature} by PHQ8 Category')
    plt.axhline(y=value, color='red', linestyle='--', linewidth=2)
    plt.xlabel('PHQ8 Category')
    plt.ylabel(feature)
    plt.tight_layout()
    # Save the modified plot
    feature_name = feature_name.lower()
    # D:\Aayush\Aayush\Mental Health Project\OneDrive_2024-05-14\mental health\static\{feature_name}_{recording_index}.png
    image_path = f'D:/Aayush/Aayush/Mental Health Project/OneDrive_2024-05-14/mental health/static/{feature_name}_{recording_index}.png'
    image_path = image_path.replace(" ","")
    plt.savefig(image_path)
    plt.close()



##use plotly


def plot_speaking_rate(speaking_rate, recording_index, line_value):
    # Open the existing plot
    # D:\Aayush\Aayush\Mental Health Project\OneDrive_2024-05-14\mental health\static\speaking_rate_plot_{recording_index}.png
    image_path = f'D:/Aayush/Aayush/Mental Health Project/OneDrive_2024-05-14/mental health/static/speaking_rate_plot_{recording_index}.png'
    img = mpimg.imread(image_path)
    
    # Plot the existing image
    plt.imshow(img)
    
    # Add horizontal line at specified value
    plt.axhline(y=line_value, color='red', linestyle='--', linewidth=2)
    
    # Save the modified plot
    plt.savefig(image_path)
    plt.close()

def plot_percent_pause_time(percent_pause_time, recording_index, line_value):
    # Open the existing plot
    # D:\Aayush\Aayush\Mental Health Project\OneDrive_2024-05-14\mental health\static\percent_pause_time_plot_{recording_index}.png
    image_path = f'D:/Aayush/Aayush/Mental Health Project/OneDrive_2024-05-14/mental health/static/percent_pause_time_plot_{recording_index}.png'
    img = mpimg.imread(image_path)
    
    # Plot the existing image
    plt.imshow(img)
    
    # Add horizontal line at specified value
    plt.axhline(y=line_value, color='red', linestyle='--', linewidth=2)
    
    # Save the modified plot
    plt.savefig(image_path)
    plt.close()

@app.route("/")
def index():
    global recordings_with_emotion
    # Fetch recordings with emotion details
    recordings_with_emotion = get_recordings_and_predict(0, 5)
    return render_template('index.html', recordings_with_emotion=recordings_with_emotion)
import plotly.graph_objs as go
import plotly.express as px

def plot_master_plotly(value, feature_name, recording_index, ind):
    # Load data for plotting
    #  D:\Aayush\Aayush\Mental Health Project\OneDrive_2024-05-14\mental health\audio_features_EATD.csv
    # if ind == 4:
        # print("Hello from f0_stdev")
    df = pd.read_csv("D:/Aayush/Aayush/Mental Health Project/OneDrive_2024-05-14/mental health/new_csv.csv")
    df['PHQ8 Category'] = pd.cut(df['SDS Score'], bins=[0, 50, float('inf')], labels=['Normal', 'Needs Attention'])
    
    features_to_plot = ['Pitch Variation', 'Speaking Rate', 'Percent Pause Time', 
                        'F0 Mean', 'F0 Standard Deviation', 'HNR', 
                        'Local Jitter', 'Local Shimmer']
    feature = features_to_plot[ind]
    # if ind == 4:
        # print(feature)
    
    # Create box plot
    
    fig = px.box(df, x='PHQ8 Category', y=feature, points=False)
    
    # Add line for value
    fig.add_shape(
        type='line',
        x0=0,
        y0=value,
        x1=1,
        y1=value,
        line=dict(color='red', width=2, dash='dash')
    )
    # if ind == 4:
        # print(fig)

    # Save HTML representation of plot
    plot_div = fig.to_html(full_html=False)
    
    # Save the HTML representation
    # D:\Aayush\Aayush\Mental Health Project\OneDrive_2024-05-14\mental health\templates\{feature_name}_{recording_index}.html
    with open(f'D:/Aayush/Aayush/Mental Health Project/OneDrive_2024-05-14/mental health/templates/{feature_name}_{recording_index}.html', 'w', encoding='utf-8') as file:
        file.write(plot_div)

@app.route("/detailed_analysis", methods=["GET", "POST"])
def detailed_analysis():
    if request.method == "POST":
        selected_recording_sid = request.form.get("recording_sid")
        selected_recording = next((recording for recording in recordings_with_emotion if recording["RecordingSID"] == selected_recording_sid), None)
        if selected_recording:
            # Call the plot functions and save the plots
            plot_master_plotly(selected_recording['PitchVariation'], 'pitch_variation', selected_recording_sid, 0)
            plot_master_plotly(selected_recording['SpeakingRate'], 'speaking_rate', selected_recording_sid, 1)
            plot_master_plotly(selected_recording['PercentPauseTime'], 'percent_pause_time', selected_recording_sid, 2)
            plot_master_plotly(selected_recording['f0_mean'], 'f0_mean', selected_recording_sid, 3)
            plot_master_plotly(selected_recording['f0_stdev'], 'f0_stdev', selected_recording_sid, 4)
            plot_master_plotly(selected_recording['hnr'], 'hnr', selected_recording_sid, 5)
            plot_master_plotly(selected_recording['localjitter'], 'localjitter', selected_recording_sid, 6)
            plot_master_plotly(selected_recording['localshimmer'], 'localshimmer', selected_recording_sid, 7)
            
            # Render the template with the selected recording and paths of the saved plot images
            return render_template('detailed_analysis.html', selected_recording=selected_recording, recordings_with_emotion=recordings_with_emotion, recording_sid=selected_recording_sid)
    
    return render_template('detailed_analysis.html', selected_recording=None, recordings_with_emotion=recordings_with_emotion)



if __name__ == "__main__":
    app.run(debug=True)
