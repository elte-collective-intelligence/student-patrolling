import imageio
import numpy as np
from pathlib import Path

def save_episode_gif(
    frames,
    path,
    fps=120
):
    """
    Save a list of RGB frames as a GIF.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    frames = [frame.astype(np.uint8) for frame in frames]

    imageio.mimsave(
        uri=str(path),
        ims=frames,
        fps=fps,
        loop=0
    )
