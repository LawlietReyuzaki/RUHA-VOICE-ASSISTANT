import pandas as pd
import os 
import shutil
from pydub import AudioSegment
less_files = []

#knn = pd.DataFrame()

i =0
Recordings = []
Lables = []
Main = []
for Class_name in os.listdir("Classes"):
    
	path_inside = "Classes/" + Class_name
		
	for audio in os.listdir(path_inside):
		Lables.append(Class_name)
		Recordings.append(path_inside+'/'+audio)
		pos = audio.find("_")
		audio = audio.replace(".wav", "")
		Main.append(audio[pos+1:])

	

#knn.apply(LabelEncoder().fit_transform)
#knn.apply(LableEncoder())
#print(knn)

print('\n\n\n\n problematic files are :', less_files, '\n\n\n\n')

import librosa
import soundfile as sf
#import numpy as np

hop_length = 512

list1 =[]
list2 = []
list3 = []
list4= []
list5= []
list6= []
MFCC_Features= 0
chromagram =0
S=0
#arr = np.array([])

problem = []
k=0
for x in Recordings:
	audio_path = x
	'''if(x.find(".mp3") != -1):
		continue'''
	try:
		print(x)
		#MFCC
		X,sample_rate = sf.read(x, dtype='float32')
		MFCC_Features = librosa.feature.mfcc(y=X, sr=sample_rate)
		MFCC_Features = [j for sub in MFCC_Features for j in sub]
		MFCC_Features = sum(MFCC_Features)/len(MFCC_Features)
		
		#chromagram
		chromagram = librosa.feature.chroma_stft(X, sr=sample_rate, hop_length=hop_length)
		chromagram = [j for sub in chromagram for j in sub]
		chromagram = sum(chromagram)/len(chromagram)
		
		#melspectrogram
		S = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=128,fmax=8000)
		S = [j for sub in S for j in sub]
		S = sum(S)/len(S)
		print("value of s:", S)
		print("This is chromagram: ", chromagram)

		#tonnetz
		t = librosa.effects.harmonic(X)
		tonnetz = librosa.feature.tonnetz(y=t, sr=sample_rate)
		t = [j for sub in t for j in sub]
		t = sum(t)/len(t)
		print("value of t:", t)
		print("This is t: ", chromagram)
		
		
	except:
		try:
			stereo_audio = AudioSegment.from_file(x,format="wav")
			mono_audios = stereo_audio.split_to_mono()
			mono = mono_audios[0].export(x,format="wav")
			print ('\n\n\n\n\n',mono,'\n\n\n\n\n\n\n')
			X,sample_rate = sf.read(x, dtype='float32')
			MFCC_Features = librosa.feature.mfcc(y=X, sr=sample_rate)
			MFCC_Features = [j for sub in MFCC_Features for j in sub]
			MFCC_Features = sum(MFCC_Features)/len(MFCC_Features)
			
			#chromagram
			chromagram = librosa.feature.chroma_stft(X, sr=sample_rate, hop_length=hop_length)
			chromagram = [j for sub in chromagram for j in sub]
			chromagram = sum(chromagram)/len(chromagram)			

			#melspectrogram
			S = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=128,fmax=8000)
			S = [j for sub in S for j in sub]
			S = sum(S)/len(S)
			print("value of s:", S)
			
			#tonnetz
			t = librosa.effects.harmonic(X)
			tonnetz = librosa.feature.tonnetz(y=t, sr=sample_rate)
			t = [j for sub in t for j in sub]
			t = sum(t)/len(t)
			print("value of t:", t)
			print("This is t: ", chromagram)
		except:
			print("Error")
			problem.append(x)
			
			
	list1.append(MFCC_Features)
	list2.append(Main[k])
	list3.append(Lables[k])
	list4.append(chromagram)
	list5.append(S)
	list6.append(t)
	
	print('\n\n\n')
	print("list1", len(list1))
	print("list2", len(list2))
	print("list3", len(list3))
	print("list4", len(list4))
	print("list5", len(list5))
	print("list6", len(list6))
	
	
	k = k+1
	print (MFCC_Features)		

from itertools import chain

try:		
	df2 = pd.DataFrame()
	df2['Lable1'] = list2
	df2['Lable2'] = list3
	df2['MFCC'] = list1
	df2['chromogram'] = list4
	df2['melspectrogram'] = list5
	df2['tonnetz'] = list6
	df2.to_csv("sound MFCC.csv")

except:
	df2 = pd.DataFrame()
	df2['Lable1'] = list2[:5500]
	df2['Lable2'] = list3[:5500]
	df2['MFCC'] = list1[:5500]
	df2['chromogram'] = list4[:5500]
	df2['melspectrogram'] = list5[:5500]
	df2['tonnetz'] = list6[:5500]
	df2.to_csv("sound MFCC.csv")
	
df3 = pd.DataFrame()
df3['Lable1'] = list2
df3['Lable2'] = list3
df3['MFCC'] = list1
df3['chromogram'] = list4
df3['melspectrogram'] = list5
df3['tonnetz'] = list6
df3.to_csv("sound MFCC2.csv")

print(df2)
print(len(problem))
for x in problem:
	print(x)
