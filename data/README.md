# Data directory

Place a locally downloaded and extracted EdNet KT2 dataset in
`data/raw/ednet-kt2/`. The repository intentionally excludes the dataset because
the uncompressed download is several gigabytes.

The loader expects one CSV per student, as distributed by EdNet. It can also read a
single compatible CSV or ZIP archive. The student ID is inferred from the filename
when the file does not contain a `user_id` column.

Expected KT2 fields from the EdNet documentation:

* `timestamp` — Unix timestamp in milliseconds
* `action_type` — `enter`, `respond`, or `submit`
* `item_id` — question or bundle identifier
* `source` — learning source such as `sprint` or `tutor`
* `user_answer` — answer choice for response actions
* `platform` — `mobile` or `web`

Raw data and generated processed data are ignored by Git.
