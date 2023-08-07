import librosa
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import IPython.display as ipd
import soundfile as sf
import struct
buff_size = 512
from fastcore.all import *
from fastai.vision.all import *

import json
import os
import pandas as pd
from io import StringIO
import streamlit as st
from audio_recorder_streamlit import audio_recorder

title = "Zeno"

st.set_page_config(page_title=title, page_icon=":robot:")
st.header(title)
st.markdown(
    f" Perform the voice exam to obtain a diagnosis.",
    unsafe_allow_html=True,
)
st.divider()
sample_rate = 44_100
def record_audio():

    audio_bytes = audio_recorder(
        text="Click to Record you Audio, say 'ahh' continously for 10 seconds",
        recording_color="#e8b62c",
        neutral_color="#6aa36f",
        icon_name="user",
        icon_size="4x",
        energy_threshold=(-1.0, 1.0),
        pause_threshold=10.0, # Number of seconds to record.,
        sample_rate  = sample_rate
    )
    if audio_bytes:    
        return audio_bytes

def calculate_prob(result):
    if result is None:
        return None
    one_prob = result[2].numpy()[0] * 100
    zero_prob = result[2].numpy()[1] * 100
    result = int(result[0])
    if result == 0:
        zero_prob
    else:
        one_prob
    output = f"Prob of having PD {zero_prob}%, Prob of healthy {one_prob}%"
    return output


def main():
    audio_bytes = record_audio()
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        st.markdown("Recorded your audio, pleaes click to process your audio")
        if st.button('Process my Audio'):
            st.write(f'Processing audio')
            # Convert the audio generator to a numpy array
            audio_generator = librosa.stream(io.BytesIO(audio_bytes), block_length=256, frame_length=2048, hop_length=2048)
            audio_array = np.concatenate(list(audio_generator))
            
            # Save the numpy array as a WAV file using soundfile
            sf.write("test.wav", audio_array, samplerate=sample_rate)
            
            # Now you can compute the MFCCs using the audio_array
            mfccs = librosa.feature.mfcc(y=audio_array, sr=sample_rate, n_mfcc=40)
            librosa.display.specshow(mfccs, sr=sample_rate, x_axis="time")
            plt.savefig(f"mfcc.png")            
            predictor = load_learner("model/zeno_model.pkl")  
            result = predictor.predict(mfccs)
            out = calculate_prob(result)    
            st.markdown(f"**Your Result {out}**")
            st.image("mfcc.png")

def test_affected():
    audio_array,sr = librosa.load("affected5.tmp")
    mfccs = librosa.feature.mfcc(y=audio_array, sr=sample_rate, n_mfcc=40)
    librosa.display.specshow(mfccs, sr=sample_rate, x_axis="time")
    plt.savefig(f"mfcc.png")
    predictor = load_learner("zeno_model.pkl")  
    result = predictor.predict(mfccs)
    out = calculate_prob(result)    
    st.markdown(f"# {out}")
    st.image("mfcc.png")

if __name__ == "__main__":
    label_func = {}
    main()

    
