"""
train.py  –  Backward-compatible entry point (forwards to main.py).

NOTE: This file is superseded by the OOP refactor.
      - Entry point    : main.py  (ExperimentRunner)
      - Training logic : utils/utilities.py  (Trainer, TODOs 8-11)
      - Data loading   : data.py  (DataManager, TODOs 6-7)
      - Models         : models.py  (TODOs 1-5)

You may run experiments via either:
  python main.py --dataset PROTEINS --model gin ...
  python train.py --dataset PROTEINS --model gin ...  # same CLI, forwards here
"""
from main import parse_args, ExperimentRunner  # noqa: F401


if __name__ == "__main__":
    args = parse_args()
    if args.seed is not None:
        args.seeds = [args.seed]
    ExperimentRunner(args).run()
