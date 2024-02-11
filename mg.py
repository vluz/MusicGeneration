import streamlit as st
import gc
import torch
import torchaudio
import time
from audiocraft.models import MusicGen
from audiocraft.data.audio_utils import i16_pcm, normalize_audio


@st.cache_resource
def loadmodel(name):
    model = MusicGen.get_pretrained(name)
    return model


def setparams(model, use_sampling, top_k, top_p, temperature, cfg_coef, duration):
    model.set_generation_params(
        use_sampling=use_sampling,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        cfg_coef=cfg_coef,
        duration=duration)
    return model


def generate(model, prompt):
    output = model.generate(descriptions=prompt, progress=True)
    return output


def purge():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()


st.title("MG Music Gen")
st.divider()
option = st.selectbox('Select a model:', ('small', 'medium', 'melody', 'large'))
prompt = st.text_input(label='Prompt:', value='90s rock song with loud guitars and heavy drums')
if st.button("Generate"):
    purge()
    model = loadmodel(option)
    model_ready = setparams(model, use_sampling=True, top_k=250, top_p=0.0, temperature=1.0, cfg_coef=3.0, duration=10)
    purge()
    with st.spinner("Generating..."):
        output = generate(model_ready, prompt)
    with st.spinner("Detaching..."):
        output = output.detach().cpu().float()[0]
    with st.spinner("Normalizing..."):
        wav = normalize_audio(wav=output, sample_rate=32000, strategy="rms")
    with st.spinner("PCM Encoding..."):
        pcm_audio = i16_pcm(wav)
    with st.spinner("Saving..."):
        now = str(time.strftime("%Y%m%d-%H%M%S"))
        filename = now + ".wav"
        torchaudio.save(filename, pcm_audio, sample_rate=32000, encoding="PCM_S", bits_per_sample=16)
    st.write("File " + filename + " created.")
    with st.spinner("Loading..."):
        with open(filename, "rb") as f:
            generation = f.read()
        st.audio(generation)
purge()

