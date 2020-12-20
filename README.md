## speech-emotion-recognition-using-self-attention
In this project we implement the paper "Improved End-to-End Speech Emotion Recognition Using Self Attention
Mechanism and Multitask Learning" Published in INTERSPEECH 2019. 

Please note that, some of the hyperparameters are changed in order to make the convergence better

NOTE: RESULTS ARE NOT AS GOOD AS THE PAPER. USE IT AT YOUR OWN RISK


## Data Processing
First we need to create a pickle file for every combination of 5-fold cross validation. 
Use dataset.py and change the paths accordingly
```
python dataset.py
```

## Training
This step trains CNN-LSTM model
```
python train_audio_only.py
```
The average accuracy is about ~53% (UW) and 52% (WA) for CNN-BLSTM. The paper reports 55%(WA) and 51% (UW) using CNN-BILSTM-ATTENTION model. 

WORK IN PROGRESS

Paper link : https://www.isca-speech.org/archive/Interspeech_2019/pdfs/2594.pdf


