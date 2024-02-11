# Music Generation
### Music Generation Using MusicGen and Audiocraft

Uses Streamlit as gui.

Meta recently released MusicGen.
<br>
It can generate short new pieces of music based on text prompts, 
<br>
which can optionally be aligned to an existing melody.

MusicGen is based on a Transformer model. 
<br>
MusicGen predicts the next section in a music sequence.

The researchers decompose the audio data into smaller components 
<br>
using Meta's EnCodec audio tokenizer. 
<br>
A single-stage model that processes tokens in parallel.
<br>
MusicGen is fast and efficient but does require a bit of VRAM to run.
<br>
Currently around 16GB of vram for the smallest model.

The researchers used 20k hours of licensed music for training. 
<br>
In particular, a internal dataset of 10000 high-quality music tracks
<br>
and as music data from Shutterstock and Pond5.

<hr>

Open a command prompt and `cd` to a new directory of your choosing:

(optional; recommended) Create a virtual environment with:
```
python -m venv "venv"
venv\Scripts\activate
```

To install do:
```
git clone https://github.com/vluz/MusicGeneration.git
cd MusicGeneration
pip install -r requirements.txt
```

Here is a tested set of requirements updated 11-02-2024:      
```
audiocraft==1.3.0a1
streamlit==1.30.0
torch==2.1.2+cu121
torchaudio==2.1.2+cu121
```

On first run it may download several models.
<br>
The GUI may be blank or unresponsive for the duration of the setup
<br>
It will take quite some time, both on reqs above and on first run.
<br>
Please allow it time to finish.
<br>
All runs after the first are then faster to load.

To run do:<br>
`streamlit run mg.py`

Gui will open on your default browser

<hr>

~~TODO: Take adavantage of caching to speed up the app~~
<br>
TODO: Use experimental garbage collect to limit memory use

Note: Do not use this for production, it's untested



