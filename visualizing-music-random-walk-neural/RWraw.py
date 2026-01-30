import os
import numpy as np
import torch
from PIL import Image, ImageDraw

class RW_Painter:
    def __init__(self, model, W, H):
        self.model = model
        self.W = W
        self.H = H
        self.canvas = Image.new("RGB", (W, H), (0, 0, 0))
        self.draw = ImageDraw.Draw(self.canvas)
        self.x = (W - 1) / 2.0
        self.y = (H - 1) / 2.0
        self.theta = 0.0

    def render(self, spec, num_frames, steps_per_frame, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        with torch.no_grad():
            for f in range(num_frames):
                spec_frame = spec[f]
                intensity = float(spec_frame.mean().item())

                step_len = 0.5 + 6.0 * intensity
                width = int(1 + 4 * intensity)
                turn_std = 0.03 + 1.2 * intensity

                for _ in range(steps_per_frame):
                    self.theta += np.random.normal(0.0, turn_std)

                    x2 = self.x + step_len * np.cos(self.theta)
                    y2 = self.y + step_len * np.sin(self.theta)
                    x2 = float(np.clip(x2, 0, self.W - 1))
                    y2 = float(np.clip(y2, 0, self.H - 1))

                    x_norm = (self.x / (self.W - 1)) * 2.0 - 1.0
                    y_norm = (self.y / (self.H - 1)) * 2.0 - 1.0
                    xy = torch.tensor([[x_norm, y_norm]], dtype=torch.float32)

                    inp = torch.cat([xy, spec_frame.unsqueeze(0)], dim=1)
                    rgb = self.model(inp)[0].cpu().numpy()
                    color = tuple((rgb * 255).astype(np.uint8))

                    self.draw.line([(self.x, self.y), (x2, y2)], fill=color, width=width)
                    self.x, self.y = x2, y2

                frame_path = os.path.join(output_dir, f"{f:06d}.png")
                self.canvas.save(frame_path)

                if (f + 1) % 50 == 0:
                    print(f"Saved {f + 1}/{num_frames} frames")



