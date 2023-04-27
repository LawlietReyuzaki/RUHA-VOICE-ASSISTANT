import os
import uuid
from flask import Flask, flash, request, redirect
import librosa
import pickle
import soundfile as sf
import pandas as pd
import os
import sys

UPLOAD_FOLDER = 'files'
hop_length = 512


#importing model
with open('RUHA.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def root():
    return app.send_static_file('index.html')


@app.route('/save-record', methods=['POST'])
def save_record():
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    file_name = "recording.mp3"
    #C:\\Users\\A1D\\Desktop\\RECORDER
    full_file_name = os.path.join('C:\\Users\\A1D\\Desktop\\RECORDER', file_name)
    file.save(full_file_name)
    
    #sample_rate = 44100
    #channels = 2
    #audio_data,_ = sf.read(file)
    #sf.write('this.mp3', audio_data, sample_rate, subtype='PCM_16')  # save the audio data to a file    

    return '<h1>Success</h1>'

@app.route('/get-data', methods=['GET'])
def get_flask_response():
   # Retrieve some data to return to frontend
   response = "Hello, frontend!"
   audio_file = "C:\\Users\\A1D\\Desktop\\RECORDER\\recording.mp3"
   #audio_file = AudioSegment.from_file(audio)

   X,sample_rate = librosa.load(audio_file, dtype='float32',sr=44100, mono=False)
   X=X.T
   
   #MFCC
   #X,sample_rate = sf.read(audio_file, dtype='float32')
   MFCC_Features = librosa.feature.mfcc(y=X, sr=sample_rate)
   MFCC_Features = [j for sub in MFCC_Features for j in sub]
   MFCC_Features = sum(MFCC_Features)/len(MFCC_Features)

   #chromagram

   chromagram = librosa.feature.chroma_stft(S=X, sr=sample_rate, hop_length=hop_length)
   chromagram = [j for sub in chromagram for j in sub]
   chromagram = sum(chromagram)/len(chromagram)

   #melspectrogram
   S = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=128,fmax=8000)
   S = [j for sub in S for j in sub]
   S = sum(S)/len(S)


   df = pd.DataFrame([[MFCC_Features,chromagram,S]])

   label = model.predict(df)
    
    
   # Remove the audio file from disk
   #os.remove('audio.wav')    


   return label[0]
    
   #return "message"


if __name__ == '__main__':
    app.run()      