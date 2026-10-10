"""Export small weight-only snapshots from the (large, local-only) Fig. 5 checkpoints.

Snapshots are NOT tracked by git (the repo .gitignore excludes *.pt on purpose); copy them yourself if you want
them on another machine.

    python export_weights.py [runs/fig5]
Writes <tag>.weights.pt = {online (float16), episode, scheme, seed}. fig5_convergence.py resumes from it
when the .ckpt is gone (fresh Adam, empty replay, fresh FSMC seeded from the episode) -- a fallback if the
cloud container is recycled.
"""
import pickle
import sys
from pathlib import Path

import torch

import common  # noqa: F401  (puts the Liu packages on sys.path so the checkpoint unpickles)
import factored_dqn  # noqa: F401


def main() -> None:
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/fig5")
    for ck_path in sorted(d.glob("*.ckpt")):
        ck = pickle.loads(ck_path.read_bytes())
        tag = ck_path.stem
        snap = {"online": {k: v.half() for k, v in ck["online"].items()}, "episode": ck["episode"], "tag": tag}
        torch.save(snap, d / f"{tag}.weights.pt")
        print(f"{tag}: episode {ck['episode']}, {(d / f'{tag}.weights.pt').stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
