"""Every path used by the scripts, relative to the repository root.

Set TPD_ROOT to run with data and results kept outside the repository.
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("TPD_ROOT", Path(__file__).resolve().parents[1]))

DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures" / "output"
PAPER_FIGURES = ROOT / "paper_reference" / "figures"

#: the raw SNAP record, `data/README.md` says where it comes from
EMAIL_TXT = DATA / "email-Eu-core-temporal.txt"
EMAIL_GZ = DATA / "email-Eu-core-temporal.txt.gz"


def email_record():
    """The raw record, uncompressed if it has been unpacked, gzipped otherwise."""
    if EMAIL_TXT.exists():
        return EMAIL_TXT
    if EMAIL_GZ.exists():
        return EMAIL_GZ
    raise FileNotFoundError(
        f"neither {EMAIL_TXT} nor {EMAIL_GZ} is present - see data/README.md")

#: stage 1, daily graphs G_n  (pipeline/build_daily_graphs.py)
DAILY_GRAPHS = RESULTS / "daily_graphs.pkl"
#: stage 2, birth-edge-labelled barcodes and TPD  (pipeline/tpd_daily.py)
TPD_DAILY = RESULTS / "tpd_daily.pkl"
#: stage 3, the Wasserstein baseline  (pipeline/baseline_wasserstein.py)
WASSERSTEIN = RESULTS / "wasserstein_baseline.pkl"
#: stage 4, the three series in one table  (pipeline/make_series_weekend.py)
SERIES_CSV = RESULTS / "series_weekend.csv"

#: Section 7.3, the chain d_alpha  (experiments/run_alpha_sweep.py)
ALPHA_SWEEP = RESULTS / "alpha_sweep_unit.pkl"
#: Section 7.4, recency-weighted contacts  (experiments/run_recency_sweep.py)
RECENCY_SWEEP = RESULTS / "recency_sweep.pkl"
#: Appendix C.4, the table of baselines  (experiments/friday_baseline_table.py)
FRIDAY_TABLE = RESULTS / "tab_friday_baselines.tex"


def ensure_dirs() -> None:
    for d in (RESULTS, FIGURES):
        d.mkdir(parents=True, exist_ok=True)
