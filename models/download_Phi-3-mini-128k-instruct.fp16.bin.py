# Downloader to get the model file for Phi-3-mini-128k-instruct.fp16.bin
from urllib.request import urlretrieve
import os

if os.path.exists("Phi-3-mini-128k-instruct.fp16.bin"):
    print("Phi-3-mini-128k-instruct.fp16.bin already exists")
    exit(0)
print("Downloading Phi-3-mini-128k-instruct.fp16.bin")
url = "https://huggingface.co/PrunaAI/Phi-3-mini-128k-instruct-GGUF-Imatrix-smashed/resolve/main/Phi-3-mini-128k-instruct.fp16.bin?download=true"
dest = "Phi-3-mini-128k-instruct.fp16.bin"
urlretrieve(url, dest)
print("Downloaded Phi-3-mini-128k-instruct.fp16.bin")
