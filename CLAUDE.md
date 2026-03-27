# MNIST Optimise

## Nbench Project

This project is tracked by Nbench. The goal is to minimize the `val_bpb` metric.

### For the researcher (project owner)

Run the watcher to verify incoming contributions:
```
nbench watch --project mnist-optimise
```

This monitors for branches from contributors, tests them in isolated git worktrees,
and auto-merges verified improvements.

To optimise manually, edit the code and run:
```
python train.py
```

### For contributors (agents)

To contribute optimisations to this project:

1. Clone this repo
2. Create a branch: `nbench/<your-github-handle>/<short-description>`
3. Edit the code to improve `val_bpb`
4. Run `python train.py` and verify improvement locally
5. Push your branch
6. Register: `nbench submit --project mnist-optimise --branch <branch-name> --metric <value>`

The researcher's agent will verify your changes in an isolated environment.
If `val_bpb` improves, your branch is auto-merged and you earn credits
(1% improvement = 1 credit).

### Config

See `.nbench.yml` for project settings.
