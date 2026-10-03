"""Render the documentation site with pdoc.

    uv run --extra playground --with pdoc python docs/build.py --out site

The site serves the GIFs that `docs/tasks.md` shows from `gifs/`, copied from `--gifs`.
`docs/gifs` holds them in Git LFS, which clones skip; fetch them with
`git lfs pull --include "docs/gifs/**" --exclude ""`. `.github/workflows/docs.yml`
builds and publishes the site on every push to `main`.
"""

import argparse
import shutil
from pathlib import Path

import pdoc
import pdoc.docstrings
import pdoc.render


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gifs", type=Path, default=Path(__file__).parent / "gifs")
    parser.add_argument("--out", type=Path, default=Path("site"))
    args = parser.parse_args()
    modules = ["jax_gym"]
    try:
        import jax_gym.playground  # noqa: F401
    except ImportError:
        print("mujoco_playground is not installed; skipping jax_gym.playground")
    else:
        modules.append("jax_gym.playground")
    # The site links the GIFs rather than inlining them.
    pdoc.docstrings.embed_images = lambda docstring, source_file: docstring
    pdoc.render.configure(docformat="restructuredtext", math=True, search=True)
    pdoc.pdoc(*modules, output_directory=args.out)
    shutil.copytree(args.gifs, args.out / "gifs", dirs_exist_ok=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
