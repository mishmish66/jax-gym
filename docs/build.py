"""Render the documentation site with pdoc.

    uv run --extra playground --with pdoc python docs/build.py --gifs <dir> --out site

`--gifs` holds the task GIFs that `docs/tasks.md` shows, named as it names them; the
site serves them from `gifs/`.
"""

import argparse
import os
import shutil
from pathlib import Path

import pdoc
import pdoc.render


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gifs", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("site"))
    args = parser.parse_args()
    modules = ["jax_gym"]
    try:
        import jax_gym.playground  # noqa: F401
    except ImportError:
        print("mujoco_playground is not installed; skipping jax_gym.playground")
    else:
        modules.append("jax_gym.playground")
    os.environ["PDOC_EMBED_IMAGES"] = "0"
    pdoc.render.configure(docformat="restructuredtext", math=True, search=True)
    pdoc.pdoc(*modules, output_directory=args.out)
    shutil.copytree(args.gifs, args.out / "gifs", dirs_exist_ok=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
