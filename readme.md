Run `sh run.sh`; the results will be generated in the `embedding` directory.

This will take a few minutes to produce all of the files. You can optionally comment out parts of the code (for example, the portion that writes to Parquet) to speed things up.
The `embedding` directory will end up looking like this:

```
865M    all_embeddings_columns.csv              150,528 rows 633 columns
107M    all_embeddings_columns.parquet
865M    all_embeddings_rgb_columns.csv          50,176 rows 1,899 columns
86M     all_embeddings_rgb_columns.parquet
```

Here:

* `all_embeddings_columns.csv` represents each image’s embedding as a single column; it is the result of concatenating embeddings from 633 images.
* `all_embeddings_rgb_columns.csv` represents each image’s embedding as three columns (one per R, G, and B channel); it is the result of concatenating embeddings from 633 images.
* All values are of type `float`.

Both of these files encounter the following error when compressing:

For `all_embeddings_columns.csv`:

```
Failed:         Negative index.
Exprsn:         n_exceptions >= 0 && n_exceptions <= 1023
Values:          
/home/tangyuxin/FastLanes/src/expression/slpatch_operator.cpp:131
Aborted (core dumped)
```

For `all_embeddings_rgb_columns.csv`:

```
Failed:         ABORTED.
Exprsn:         false
Values:         UNREACHABLE
/home/tangyuxin/FastLanes/src/wizard/wizard.cpp:1174
Aborted (core dumped)
```
