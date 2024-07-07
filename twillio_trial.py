# Download the helper library from https://www.twilio.com/docs/python/install
import os
from twilio.rest import Client
from time import sleep
from twilio.twiml.voice_response import VoiceResponse

# Set environment variables for your credentials
# Read more at http://twil.io/secure

account_sid = "ACf6824a516890c2af9fe97dd44e79e887"
auth_token = "8d0d5f80d101ea6104f8e61a64f43647"
client = Client(account_sid, auth_token)

def create_twiml():
    response = VoiceResponse()

    # Prompt for name
    response.say(" Hello. Please introduce yourself in about 30 seconds. Thank You. ", voice='amy', language='en-US')
    # response.record(max_length=10, action='/process_name', method='POST')

    # Wait for 10 seconds

    # response.pause(length=10)

    # Introduce yourself
    # response.say("Thank you. My name is ChatGPT. I'm a conversational AI developed by OpenAI.", voice='alice', language='en-US')

    # Wait for 30 seconds
    response.pause(length=30)

    # Final message
    # response.say("You can hang up now. Goodbye!", voice='alice', language='en-US')

    return str(response)

# # Create TwiML and initiate the call
Twiml = create_twiml()
# call = client.calls.crea

call = client.calls.create(record=True,
  twiml=Twiml,
  to="+918446616715",
  from_="+1 412 324 4721"
)

print(call.sid)
# print(recording.sid)

# # # Download the helper library from https://www.twilio.com/docs/python/install
# # import os
# # from twilio.rest import Client

# # # Find your Account SID and Auth Token at twilio.com/console
# # # and set the environment variables. See http://twil.io/secure


# account_sid = "AC4574b4046e97d6dae0a9ae00925abac7"
# auth_token = "62293419d54381f204214f291ea0f789"
# # # # account_sid = os.environ['TWILIO_ACCOUNT_SID']
# # # # auth_token = os.environ['TWILIO_AUTH_TOKEN']
# # # client = Client(account_sid, auth_token)

# # # recording = client.recordings('REXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX').fetch()

# # # print(recording.call_sid)


# # # import requests
# # # recording_sid = recording
# # # file_extension = "mp3"

# # # # Twilio API endpoint for fetching a recording
# # # url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Recordings/{recording_sid}.{file_extension}"

# # # # Set up authentication using your Twilio Account SID and Auth Token
# # # auth = (account_sid, auth_token)

# # # # Make the GET request
# # # response = requests.get(url, auth=auth)

# # # if response.status_code == 200:
# # #     # Successful request, you can handle the recording content here
# # #     with open(f"downloaded_recording.{file_extension}", "wb") as file:
# # #         file.write(response.content)
# # #     print("Recording downloaded successfully!")
# # # else:
# # #     # Handle errors
# # #     print(f"Error: {response.status_code} - {response.text}")


# # from twilio.twiml.voice_response import Record, VoiceResponse, Say

# # response = VoiceResponse()
# # response.say(
# #     'Please leave a message at the beep.\nPress the star key when finished.'
# # )
# # response.record(
# #     action='http://foo.edu/handleRecording.php',
# #     method='GET',
# #     max_length=20,
# #     finish_on_key='*'
# # )
# # response.say('I did not receive a recording')

# # print(response)

# from twilio.rest import Client

# # Replace these placeholders with your Twilio Account SID, Auth Token, and Twilio phone numbers
# from_phone_number = "+18145643847"

# to_phone_number = "+917014153288"

# # Create a Twilio client
# client = Client(account_sid, auth_token)

# # # Make a call and record it
# # call = client.calls.create(
# #     to=to_phone_number,
# #     from_=from_phone_number,
# #     url="http://demo.twilio.com/docs/voice.xml",  # replace with your own TwiML URL
# #     method="GET",
# #     record=True
# # )

# # print(f"Call SID: {call.sid}")
# import base64
# import wave

# # Wait for the call to complete (you may need to adjust the time based on your use case)
# input("Press Enter to fetch the recording...")

# # Fetch the recording SID from the completed call
# # call = client.calls(call.sid).fetch()
# recording_sid = "RE9108e46fb6eb01026017a63a5eaa1084"

# # Download the recording as an MP3
# # recording_url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Recordings/{recording_sid}.wav"
# page_url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Recordings.json"


# recording_response = client.http_client.request("GET", page_url, auth=(account_sid, auth_token))

# if recording_response.status_code == 200:
#     # Decode the base64-encoded string to bytes

#     # Write the bytes to a WAV file
#     with open("downloaded_recording.json", "w") as file:
#         file.write(recording_response.content)

#     print("Recording downloaded successfully!")
# else:
#     print(f"Error downloading recording: {recording_response.status_code} - {recording_response.text}")
