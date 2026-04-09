"""
Compatibility entrypoint for the autoresearch split.

The mutable research surface now lives in `train.py`.
The immutable runtime, evaluation, and artifact logic now lives in `prepare.py`.
"""

from train import main


if __name__ == "__main__":
    main()
