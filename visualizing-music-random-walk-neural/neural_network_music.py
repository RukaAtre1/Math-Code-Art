import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
from IPython.display import display
import matplotlib.pyplot as plt
import torchaudio
import os
import subprocess
import shutil

# --------------------------
# 1) Network
# --------------------------
class XYSpecNet(nn.Module):
    def __init__(self, spec_dim=64, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 + spec_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 3),
            nn.Sigmoid()
        )

    def forward(self, xy_spec):
        return self.net(xy_spec)

# --------------------------
# 2) Parameters
# --------------------------
H, W = 256, 256
hidden = 64
latent_size = 64
target_fps = 30
target_img_path = "target.png"

# --------------------------
# 3) Coordinate grid (normalized)
# --------------------------
xs = torch.linspace(-1, 1, W)
ys = torch.linspace(-1, 1, H)
xx, yy = torch.meshgrid(xs, ys, indexing='xy')
coords = torch.stack([xx.flatten(), yy.flatten()], dim=1)

# ============================================================
# Helper functions (remove redundancy)
# ============================================================
def make_mel_spec(audio_file, latent_size, target_fps):
    """Load audio_file and return mel spectrogram with shape [T, latent_size]."""
    audio, sr = torchaudio.load(audio_file, normalize=True)
    audio = audio[0]  # first channel

    hop_length = int(sr / target_fps)

    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=hop_length * 2,
        n_mels=latent_size,
        hop_length=hop_length,
        win_length=hop_length * 2
    )

    spec = transform(audio).T  # [T, latent_size]
    return spec

def normalize_spec(spec, spec_min=None, spec_max=None):
    """Apply log + min-max normalize to [0,1]. If min/max not provided, compute them."""
    spec = torch.log(spec + 1e-9)

    if spec_min is None:
        spec_min = spec.min()
    if spec_max is None:
        spec_max = spec.max()

    spec = (spec - spec_min) / (spec_max - spec_min + 1e-8)
    return spec, spec_min, spec_max

def make_model_input(coords, spec_frame, H, W):
    """coords: [H*W,2], spec_frame: [latent_size] -> inp: [H*W, 2+latent_size]"""
    music_vec = spec_frame.unsqueeze(0).repeat(H * W, 1)  # [H*W, latent_size]
    inp = torch.cat([coords, music_vec], dim=1)           # [H*W, 2+latent_size]
    return inp

def frames_to_video_with_audio(
    output_dir,
    output_name,
    audio_file,
    target_fps
):
    """
    Turn frames_%06d.png into a video, add audio, and clean up.
    Returns final video path.
    """

    print("Creating video...")
    output_video = f"{output_name}.mp4"
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-framerate", str(target_fps),
        "-i", os.path.join(output_dir, "%06d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_video
    ]
    subprocess.run(ffmpeg_cmd, check=True, capture_output=True)

    print("Adding audio...")
    output_final = f"{output_name}_final.mp4"
    ffmpeg_audio_cmd = [
        "ffmpeg", "-y",
        "-i", output_video,
        "-i", audio_file,
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        output_final
    ]
    subprocess.run(ffmpeg_audio_cmd, check=True, capture_output=True)

    shutil.rmtree(output_dir, ignore_errors=True)
    os.remove(output_video)

    print(f"✓ Video saved: {output_final}\n")
    return output_final


# --------------------------
# STAGE 1: Train on target image with ALL spectrogram frames
# --------------------------
print("\n=== Stage 1: Training network on reference image ===")

audio_path = "audio.wav"
raw_spec = make_mel_spec(audio_path, latent_size, target_fps)
spectrogram, spec_min, spec_max = normalize_spec(raw_spec)

print(f"Spectrogram shape: {spectrogram.shape}")
print(f"Spectrogram range: {spectrogram.min():.6f} to {spectrogram.max():.6f}")
print(f"Mean: {spectrogram.mean():.6f}, Std: {spectrogram.std():.6f}")

# Load target image
target_img = Image.open(target_img_path).convert("RGB").resize((W, H))
target_rgb = np.asarray(target_img, dtype=np.float32) / 255.0
target_tensor = torch.from_numpy(target_rgb.reshape(-1, 3))

model = XYSpecNet(spec_dim=latent_size, hidden=hidden)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

train_frame_indices = np.random.choice(len(spectrogram), size=min(50, len(spectrogram)), replace=False)
train_specs = spectrogram[train_frame_indices]  # [num_train_frames, latent_size]

for step in range(1000):
    optimizer.zero_grad()

    idx = np.random.randint(0, len(train_specs))
    spec_frame = train_specs[idx]

    inp_train = make_model_input(coords, spec_frame, H, W)

    pred = model(inp_train)
    loss = loss_fn(pred, target_tensor)
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"Step {step}, loss={loss.item():.6f}")

print("Training complete!")
with torch.no_grad():
    trained_img = model(inp_train).reshape(H, W, 3).numpy()
    display(Image.fromarray((trained_img * 255).astype(np.uint8)))

ckpt_path = "xy_specnet_ckpt.pt"
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "spec_min": spec_min,
        "spec_max": spec_max,
        "H": H,
        "W": W,
        "latent_size": latent_size,
        "hidden": hidden,
        "target_fps": target_fps,
    },
    ckpt_path
)
print(f"Saved checkpoint: {ckpt_path}")

    
# --------------------------
# STAGE 2: Generate frames with ALL audio files
# --------------------------
def generate_video_from_audio(audio_file, output_name):
    print(f"\n=== Generating video for {audio_file} ===")

    raw_spec = make_mel_spec(audio_file, latent_size, target_fps)

    # Apply SAME normalization as training
    spec, _, _ = normalize_spec(raw_spec, spec_min=spec_min, spec_max=spec_max)

    print(f"Spectrogram shape: {spec.shape}")

    output_dir = f"frames_{output_name}"
    os.makedirs(output_dir, exist_ok=True)

    num_frames = spec.shape[0]

    with torch.no_grad():
        for f in range(num_frames):
            spec_frame = spec[f]
            inp_frame = make_model_input(coords, spec_frame, H, W)
            rgb = model(inp_frame).reshape(H, W, 3).cpu().numpy()

            frame_path = os.path.join(output_dir, f"{f:06d}.png")
            img = Image.fromarray((rgb * 255).astype(np.uint8))
            img.save(frame_path)

            if (f + 1) % 50 == 0:
                print(f"  Saved {f + 1}/{num_frames} frames")

    frames_to_video_with_audio(
        output_dir=output_dir,
        output_name=output_name,
        audio_file=audio_file,
        target_fps=target_fps
    )
