# Data

`email-Eu-core-temporal.txt.gz` is the **email-Eu-core-temporal** network of the
Stanford Large Network Dataset Collection (SNAP):

> <https://snap.stanford.edu/data/email-Eu-core-temporal.html>

It records e-mail between members of a large European research institution.
Each line is `sender receiver timestamp`, the timestamp in seconds from the
start of the record, which begins on Monday 2003-10-20.

```
lines (e-mails)         332,334
days                    804          (Section 7 uses the first 525)
self-loops                    0
sha256 of the unpacked file
    1835c94dc6685dcf2a614a1b6a0275b323dd4002f7aa111fbe9f76db8864391f
```

The file is included so that a rerun needs no network access; the scripts read
the gzipped copy directly.  To fetch it instead:

```bash
curl -LO https://snap.stanford.edu/data/email-Eu-core-temporal.txt.gz
sha256sum <(gzip -dc email-Eu-core-temporal.txt.gz)
```

Please cite the data set as SNAP asks:

> A. Paranjape, A. R. Benson, J. Leskovec.
> *Motifs in Temporal Networks.*
> Proceedings of the Tenth ACM International Conference on Web Search and
> Data Mining, 2017.
