# Fresh-clone validation

Validated on 2026-08-15 from commit `a6e6720` on branch
`agent/daf-restormer-accuracy-model`.

Procedure and result:

1. Cloned the remote branch into a new temporary directory.
2. Ran `git lfs fsck`: passed; the selected checkpoint and output arrays were materialized.
3. Created a new Python 3.12.9 environment.
4. Installed `requirements.txt` from scratch: 31 resolved packages, including PyTorch 2.6.0.
5. Ran the standalone evaluator with only `--input`, `--output`, and a one-image batch-size override. The default eight-view model loaded automatically and wrote a 256x256 `float32` NPY result on CPU.
6. Ran the full test suite: 15 passed.
7. Counted the tracked restored submission arrays: 400.

Limitation: the clone used an authenticated GitHub credential because the
organization repository is still private. Anonymous-clone validation is blocked
until a repository administrator changes the visibility to public.
