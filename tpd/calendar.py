"""The calendar convention of the e-mail record.

Day n of the record is ``floor(timestamp / 86400)`` in the time origin of the
data set itself; the record starts on 2003-10-20, a Monday.  The weekly
cycle of the record, the days without e-mail in late December 2003 and 2004,
and the busiest days of the first 525, 14-17 June 2004, right after the date
that Hajij et al. (2018) associate with the European Parliament election
results, all agree with this calendar; experiments/week_statistics.py
prints the check.  Section 7 uses the first 525 days, n in {0, ..., 524}.  The
time zone of the original timestamps is not documented.
"""
import datetime

import numpy as np

N_DAYS = 525
DAY_ZERO = datetime.date(2003, 10, 20)      # a Monday
WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                 "Saturday", "Sunday"]


def weekday_index(n_days=N_DAYS):
    """0 = Monday, ..., 6 = Sunday, for each day of the record."""
    return np.array([(DAY_ZERO + datetime.timedelta(days=t)).weekday()
                     for t in range(n_days)])


def weekday_names(n_days=N_DAYS):
    return [WEEKDAY_NAMES[i] for i in weekday_index(n_days)]


def weekend_mask(n_days=N_DAYS):
    """True on Saturdays and Sundays: 150 of the first 525 days."""
    return np.isin(weekday_index(n_days), [5, 6])
