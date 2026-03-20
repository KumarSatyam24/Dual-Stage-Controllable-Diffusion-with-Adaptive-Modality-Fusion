import os
import requests
from tqdm import tqdm

url = "https://huggingface.co/DrRORAL/ragaf-diffusion-checkpoints/resolve/main/stage2/epoch_10.pt"
save_dir = "/root/checkpoints/stage2"
save_path = os.path.join(save_dir, "epoch_10.pt")

os.makedirs(save_dir, exist_ok=True)

response = requests.get(url, stream=True)
total_size = int(response.headers.get('content-length', 0))

with open(save_path, "wb") as f, tqdm(
    desc="Downloading",
    total=total_size,
    unit='B',
    unit_scale=True,
    unit_divisor=1024,
) as bar:
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            f.write(chunk)
            bar.update(len(chunk))

print(f"✅ Download completed: {save_path}")