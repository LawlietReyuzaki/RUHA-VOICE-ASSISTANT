from flask import Flask, render_template, request
import librosa
import pickle
import soundfile as sf
import pandas as pd
import os
import sys

#importing model
with open('C:\\Users\\A1D\\Desktop\\RUHA VOICE ASSISTANT\\FLASK\\RUHA.pkl', 'rb') as f:
    model = pickle.load(f)
    
label = "Messasge"

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload-audio', methods=['POST'])
def process():
    # Get the audio file from the request
    
    #for filename, file in request.files.items():
    #    name = request.FILES[filename].name
    #    with open('responses.txt', 'w') as f:
    #    	f.write(name)


    #auth_header = request.headers.get('Authorization')
    
    audio_file = request.data

    with open('audio.wav', 'wb') as f:
    	f.write(audio_file)

    #audio_file = request.get_data()
    
    #audio_file = files.get()
    
    #string = "msg"
    #with open('temp.txt', 'w') as f:
    #    f.write(string)
    
    #f'C:\\Users\\A1D\\Desktop\\RUHA VOICE ASSISTANT\\FLASK\\{audio_file}'
    # Save the audio file to disk
    #with open(os.path.abspath('file.wav'), 'wb') as f:
    #    f.write(audio_file)

    '''#MFCC
    X,sample_rate = sf.read(audio_file, dtype='float32')
    MFCC_Features = librosa.feature.mfcc(y=X, sr=sample_rate)
    MFCC_Features = [j for sub in MFCC_Features for j in sub]
    MFCC_Features = sum(MFCC_Features)/len(MFCC_Features)

    #chromagram
    chromagram = librosa.feature.chroma_stft(X, sr=sample_rate,hop_length=hop_length)
    chromagram = [j for sub in chromagram for j in sub]
    chromagram = sum(chromagram)/len(chromagram)
    
    #melspectrogram
    S = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=128,fmax=8000)
    S = [j for sub in S for j in sub]
    S = sum(S)/len(S)
    print("MFCC:",MFCC_Features)
    print("melspectrogram:", S)
    print("chromagram: ", chromagram)

    
    df = pd.DataFrame([[MFCC_Features,chromagram,S]])
    
    label = model.predict(df)
    
    
    # Remove the audio file from disk
    #os.remove('audio.wav')'''
    return "Audio processed!"

@app.route('/get-data', methods=['GET'])
def get_flask_response():
    # Retrieve some data to return to frontend
    response = "Hello, frontend!"
    return response 
    #return jsonify(response)  


if __name__ == '__main__':
    #app.run(debug=True)
    app.run(debug = True)

