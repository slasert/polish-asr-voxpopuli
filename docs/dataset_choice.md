# Dataset choice for Polish (checked 2026-11-20)

| Candidate | License | Hours (pl) | Speakers | Speaker IDs | Transcript source | Domain | Redistribution in public repo |
|---|---|---|---|---|---|---|---|
| **VoxPopuli pl** (`facebook/voxpopuli`) — **chosen** | CC0-1.0 | train: 6 parquet files, ~18.6 GB of 32-bit audio (~80 h, estimated from file size); validation 4.95 h; test 4.93 h (measured) | yes | **yes** (`speaker_id`, gender) | official European Parliament verbatim reports (human), aligned by the dataset authors | parliamentary / political speech | **yes** (CC0) |
| Common Voice pl | CC0-1.0 data, but download requires accepting Mozilla's terms (no re-identification) | not checked | many | yes (client id) | crowd-read prompts, crowd-validated | read speech, general | unclear because of download terms; also the most likely dataset for other groups |
| FLEURS pl (`google/fleurs`) | CC-BY-4.0 | ~10 h (FLEURS paper figure, not re-checked) | unknown | **no** speaker id field | human-read FLoRes sentences | read Wikipedia-style text | yes (attribution) |
| BIGOS / PELCRA corpora | mixed, several non-commercial / no-redistribution | varies | varies | varies | mixed | mixed | no or unclear |

**Why VoxPopuli:** (1) it allows redistribution (CC0), so raw and cleaned audio can live in the
repository; (2) it has speaker IDs, so speaker-disjoint splits are possible; (3) human
transcripts; (4) a specialized domain (parliamentary speech: legal/political vocabulary,
named entities, procedure terms), which is the domain bonus; (5) Common Voice is the
default choice for many groups, so VoxPopuli lowers the risk of a dataset clash.

**What we use:** the official `validation` + `test` parquet files of the `pl` config
(3,522 clips, 9.88 h, 102 distinct speakers) as our pool. The ~18.6 GB `train` part does
not fit in a GitHub repository. We re-split the pool by speaker ourselves (`scripts/03_make_splits.py`),
because the official validation and test sets share 35 speakers.
