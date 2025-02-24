#pip install streamlit mne scipy pandas matplotlib

import streamlit as st
import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt
import mne

#Title
st.title("EEG Interactive Analysis")

upload_file= st.file_uploader("Upload your eeg file (.csv)", type=["csv"])

if upload_file:
    df = pd.read_csv(upload_file)
    st.success("File Uploaded Successfully!")

    #eeg channels
    eeg_channels = df.columns[:-1]
    eye_state_column = "eye_state" #0:close 1:open

    #choose a channel
    selected_channel = st.selectbox("Select a EEG channel", eeg_channels)
    df_columns = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    #df=df.rename(columns=channel_mapping)

    channel_mapping = {
        'AF3': 'AF3',
        'F7': 'F7',
        'F3': 'F3',
        'FC5': 'FC5',
        'T7': 'T7',
        'P': 'P3',  # 'P' potrebbe non essere nel montaggio 10-20, quindi lo mappiamo su 'P3'
        'O1': 'O1',
        'O2': 'O2',
        'P8': 'P4',  # 'P8' mappato su 'P4'
        'T8': 'T8',
        'FC6': 'FC6',
        'F4': 'F4',
        'F8': 'F8',
        'AF4': 'AF4'
    }

    df = df.rename(columns=channel_mapping)

    #select_eye_state
    select_eye_state = st.radio("Filter for eyes status:", ["All","Close","Open"] )
    if select_eye_state == "Open":
        df = df[df[eye_state_column]==1]
    elif select_eye_state == "Close":
        df = df[df[eye_state_column]==0]

    #filtering data
    fs = 256
    f_lowcut = st.slider("low cut filter (Hz)", min_value=0, max_value=30, value=1, step=1)
    f_highcut = st.slider("high cut filter (Hz)", min_value=30, max_value=100, value=50, step=1)
    notch = st.checkbox("Active Notch Filter 50Hz")


    def butter_bandpass_filter(data, f_lowcut, f_highcut, fs, order=4):
        f_nyq = fs/2
        low = f_lowcut / f_nyq
        high = f_highcut / f_nyq
        b, a = signal.butter(order, [low, high], btype='band')
        return signal.filtfilt(b, a, data)

    def notch_filter(data, f_notch, fs, quality_factor=30):
        f_nyq = fs/2
        notch = f_notch / f_nyq
        b, a = signal.iirnotch(notch, quality_factor)
        return signal.filtfilt(b, a, data)

    #apply filter
    filtered_df = df[eeg_channels].apply(lambda x:butter_bandpass_filter(x.values, f_lowcut, f_highcut, fs))
    if notch:
        filtered_df = filtered_df.apply(lambda x: notch_filter(x.values, 50, fs))

    #Select time istant
    time_idx = st.slider("t:",0,len(filtered_df)-1,100,step=1)

    eeg_values = filtered_df.iloc[time_idx].values
    info = mne.create_info(ch_names=eeg_channels.tolist(), sfreq=fs, ch_types="eeg")
    #montage_custom = mne.channels.make_dig_montage(ch_pos=df.to_dict(orient='list'))
    raw_array = mne.io.RawArray(filtered_df.T.values, info)
    montage = mne.channels.make_standard_montage("standard_1020")
    montaggio = montage.get_names()


    raw_array.set_montage(montage)
    raw_array.plot_psd()

    # Plot della Topomap
    st.subheader(f"Topomap EEG - Istante {time_idx}")
    fig, ax = plt.subplots(figsize=(6, 6))
    mne.viz.plot_topomap(eeg_values, raw_array.info, axes=ax, show=False)
    st.pyplot(fig)


"""  

    # Plot Data
    st.subheader(f"Segnale EEG - {selected_channel}")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df[selected_channel])
    ax.set_xlabel("Samples")
    ax.set_ylabel("EEG Amplitude (uV)")
    ax.set_title(f"EEG Signal for {selected_channel}")
    st.pyplot(fig)
"""



