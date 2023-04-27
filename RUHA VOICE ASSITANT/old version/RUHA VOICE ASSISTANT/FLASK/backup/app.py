from flask import Flask, render_template, request
import librosa
import pickle
import soundfile as sf
import pandas as pd
import os


#importing model
with open('C:\\Users\\A1D\\Desktop\\RUHA VOICE ASSISTANT\\FLASK\\RUHA.pkl', 'rb') as f:
    model = pickle.load(f)
    
label = "Messasge"

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    # Get the audio file from the request
    audio_file = request.files['audio']
    print("audio file: ",audio_file)
    # Save the audio file to disk
    audio_file.save('C:\\Users\\A1D\\Desktop\\RUHA VOICE ASSISTANT\\FLASK\\audio.wav')


    #MFCC
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
    #os.remove('audio.wav')

@app.route('/get-data', methods=['GET'])
def get_flask_response():
    # Retrieve some data to return to frontend
    response = "Hello, frontend!"
    return label
    #return jsonify(response)  


if __name__ == '__main__':
    #app.run(debug=True)
    app.run(debug = True)

