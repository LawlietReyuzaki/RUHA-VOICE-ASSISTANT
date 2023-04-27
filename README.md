# RUHA-VOICE-ASSISTANT
voice assistant trained on Random Forest Classifier on a small dataset. Front end created using JavaScript and FLASK framework (python)

The project consist of:

1. Python3 file for Extracting sound features of the audio file (MFCC, melspectrogram, chromatogram) -> MFCC.py 
2. csv file containing the features extracted from audio files
3. Notebooks containing training code (for 104 labels and 4 labels)
4. app.py for FLASK file (static contiaining template.html design)

FLASK APP:
1. The app allows the use to record the audio and extract it feature which are then passed to the trained model
2. The model then predicts the label
