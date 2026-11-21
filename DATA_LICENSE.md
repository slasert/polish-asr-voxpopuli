# Data license and attribution

The audio and transcripts in `data/raw/` and `data/cleaned/` (and the derived `data/splits/`)
come from **VoxPopuli**, Polish (`pl`) transcribed subset, as distributed on the Hugging Face Hub:

- Dataset: <https://huggingface.co/datasets/facebook/voxpopuli>, revision
  `42f01879c780b4a2e90ec0b4f616c2ece526e4f1`, files `pl/validation-00000-of-00001.parquet` and
  `pl/test-00000-of-00001.parquet`, downloaded 2026-11-20.
- License: **CC0 1.0 Universal** (public-domain dedication), as stated on the dataset card
  (<https://creativecommons.org/publicdomain/zero/1.0/>). The underlying recordings are European
  Parliament plenary sessions.
- Please cite: Changhan Wang, Morgane Rivière, Ann Lee, Anne Wu, Chaitanya Talnikar, Daniel Haziza,
  Mary Williamson, Juan Pino, Emmanuel Dupoux. *VoxPopuli: A Large-Scale Multilingual Speech Corpus
  for Representation Learning, Semi-Supervised Learning and Interpretation.* ACL 2021.

Changes we made: raw files are unchanged bytes (renamed: `:` replaced by `-` in file names);
cleaned files are filtered and losslessly converted to 16-bit FLAC, and some transcripts were
corrected as logged in `data/cleaning_log.csv`.

This data license is separate from the code license in `LICENSE` (MIT), which covers only our code.
