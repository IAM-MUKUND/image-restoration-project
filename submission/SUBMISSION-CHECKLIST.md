# KLA PS01 submission checklist

Generated artifacts:

- `TEAM_NAME_KLA_PS01.pptx` - editable nine-slide deck built from the official template with the instruction slide removed.
- `TEAM_NAME_KLA_PS01.pdf` - portal-ready PDF export; nine 16:9 pages visually inspected.

Repository-complete items:

- standalone `infer.py` accepting only `--input` and `--output` for the default accuracy submission;
- selected Git LFS checkpoint;
- all 400 same-name restored `.npy` files in the final predictions folder;
- output validation covering count, filenames, shape, dtype, finite values, and range;
- `train.py`, pinned `requirements.txt`, complete setup/run instructions, tests, metrics, and model card.

Owner/team actions still required before portal upload:

1. Replace `TEAM_NAME` in both artifact filenames with the registered team name.
2. Replace the clearly marked team name, academic year, college, and phone placeholders on slide 1; add or remove member rows as needed.
3. Make `IAM-MUKUND/image-restoration-project` public. The current GitHub credential has push access but not repository-admin access, so it cannot change visibility.
4. If the portal requires a prototype video rather than executable repository evidence, record/upload it and replace the second link on slide 8.
