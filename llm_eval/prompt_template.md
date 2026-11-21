You are participating in an image segmentation coding competition. Solve it end to end.

You are given the competition repository. Read the information below carefully: the README defines the task,
the data, the metric and the rules; the submission format section is binding.

### README
{{README}}

### Repository tree
{{REPO_TREE}}

### Dataset summary
{{DATASET_SUMMARY}}

### Submission format
{{SUBMISSION_FORMAT}}

### Allowed libraries
{{ALLOWED_LIBRARIES}}

### Execution constraints
- Your program will be run exactly as `python solution.py` from the repository root.
- CPU only: no GPU is available. The whole run (training + inference) must finish within {{TIME_BUDGET_MIN}} minutes.
- No internet access and no external downloads (no pretrained weights, no datasets, no pip installs).
- Use only the files under `data/public/`. Use random seed 25.
- The program must write `predictions.csv` (repository root) in the submission format above.
- Respect the competition rules in the README (e.g. the model parameter limit).

### Output format
First write your approach inside <plan> ... </plan> (data processing, model, training, post-processing, submission).
Then write the COMPLETE, self-contained content of `solution.py` inside <code> ... </code>.
Do not write anything after </code>.
