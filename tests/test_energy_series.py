"""Regression tests for Ariston energy aggregation (synthetic data)."""

from datetime import datetime
from math import nan
from types import MappingProxyType
from unittest import TestCase

from custom_components.ariston_net.energy_series import consumption_kwh


class EnergySeriesTests(TestCase):
    def setUp(self):
        self.now = datetime(2026, 10, 21, 16, 30)
        # Exactly 24 two-hour buckets, with yesterday in slots 2..13.
        # This tests the calibrated current-day tail, not sum(all 24).
        hours = [0.0] * 24
        hours[5] = 2.4  # yesterday's completed day
        hours[15] = 0.7  # today's usage
        hours[17] = 0.5
        self.samples = [
            {"k": 2, "p": 1, "v": hours},
            {"k": 2, "p": 3, "v": [1.0] * 19 + [2.4]},
            {"k": 2, "p": 4, "v": [5.0] * 8 + [31.5]},
        ]

    def test_current_month_uses_only_this_month_and_today(self):
        self.assertEqual(consumption_kwh(self.samples, 2, "current_month", self.now), 22.6)

    def test_today_excludes_yesterday_and_last_zero_bucket(self):
        self.assertEqual(consumption_kwh(self.samples, 2, "today", self.now), 1.2)

    def test_previous_month_last_completed_value(self):
        self.assertEqual(consumption_kwh(self.samples, 2, "previous_month", self.now), 31.5)

    def test_current_year_adds_completed_months(self):
        self.assertEqual(consumption_kwh(self.samples, 2, "current_year", self.now), 94.1)

    def test_first_day_month_no_prior_calendar_day_in_sum(self):
        self.assertEqual(
            consumption_kwh(self.samples, 2, "current_month", datetime(2026, 11, 1, 16, 30)),
            1.2,
        )

    def test_first_month_year_no_previous_months(self):
        self.assertEqual(
            consumption_kwh(self.samples, 2, "current_year", datetime(2027, 1, 1, 16, 30)),
            1.2,
        )

    def test_insufficient_completed_days_unknown(self):
        self.samples[1]["v"] = [2.4]
        self.assertIsNone(consumption_kwh(self.samples, 2, "current_month", self.now))

    def test_missing_hourly_series_unknown(self):
        self.assertIsNone(consumption_kwh(self.samples[1:], 2, "today", self.now))

    def test_missing_previous_day_check_unknown(self):
        self.assertIsNone(consumption_kwh(self.samples[:1], 2, "today", self.now))

    def test_mismatched_previous_day_unknown(self):
        self.samples[1]["v"][-1] = 400.0
        self.assertIsNone(consumption_kwh(self.samples, 2, "today", self.now))

    def test_zero_usage_allowed(self):
        self.samples[0]["v"] = [0.0] * 24
        self.samples[1]["v"][-1] = 0.0
        self.assertEqual(consumption_kwh(self.samples, 2, "today", self.now), 0.0)

    def test_bad_hour_bucket_count_unknown(self):
        self.samples[0]["v"] = [0.0] * 12
        self.assertIsNone(consumption_kwh(self.samples, 2, "today", self.now))

    def test_invalid_or_nan_samples_not_silently_ignored(self):
        for invalid in [None, nan, -0.1, "0", True]:
            with self.subTest(invalid=invalid):
                self.samples[0]["v"][-1] = invalid
                self.assertIsNone(consumption_kwh(self.samples, 2, "today", self.now))

    def test_different_consumption_types_isolated(self):
        self.samples.append({"k": 9, "p": 1, "v": [100.0] * 24})
        self.assertEqual(consumption_kwh(self.samples, 2, "today", self.now), 1.2)

    def test_missing_period_shape_does_not_affect_previous_month(self):
        self.samples[0]["v"] = [1.0] * 12  # an incompatible live-data shape
        self.assertIsNone(consumption_kwh(self.samples, 2, "today", self.now))
        self.assertEqual(consumption_kwh(self.samples, 2, "previous_month", self.now), 31.5)

    def test_ambiguous_zero_previous_day_fails_closed(self):
        self.samples[1]["v"][-1] = 0.0
        self.samples[0]["v"][5] = 0.0
        self.assertIsNone(consumption_kwh(self.samples, 2, "today", self.now))

    def test_other_consumption_type_same_supported_bucket_profile(self):
        for sequence in self.samples:
            sequence["k"] = 9
        self.assertEqual(consumption_kwh(self.samples, 9, "today", self.now), 1.2)

    def test_read_only_mapping_sequence_supported(self):
        read_only_sequences = tuple(MappingProxyType(s) for s in self.samples)
        self.assertEqual(
            consumption_kwh(read_only_sequences, 2, "current_month", self.now),
            22.6,
        )

    def test_one_hour_buckets_with_a_48_hour_history(self):
        hourly = [0.0] * 48
        hourly[7] = 2.4  # yesterday, 24 hourly buckets
        hourly[32] = 0.7
        hourly[37] = 0.5  # today
        self.samples[0]["v"] = hourly
        self.assertEqual(
            consumption_kwh(self.samples, 2, "today", self.now, bucket_minutes=60),
            1.2,
        )
        self.assertIsNone(
            consumption_kwh(self.samples, 2, "today", self.now, bucket_minutes=120)
        )

    def test_reporting_window_can_cover_three_days(self):
        """Older reporting buckets must not affect the current-day sum."""
        self.samples[0]["v"] = [7.5] * 12 + self.samples[0]["v"]
        self.assertEqual(
            consumption_kwh(
                self.samples, 2, "today", self.now, history_days=3
            ),
            1.2,
        )
        self.assertIsNone(
            consumption_kwh(self.samples, 2, "today", self.now)
        )

    def test_reporting_window_requires_at_least_two_days(self):
        with self.assertRaises(ValueError):
            consumption_kwh(
                self.samples, 2, "today", self.now, history_days=1
            )

    def test_bucket_duration_must_divide_day(self):
        with self.assertRaises(ValueError):
            consumption_kwh(self.samples, 2, "today", self.now, bucket_minutes=100)

    def test_huge_integer_sample_returns_unknown_without_overflow(self):
        self.samples[0]["v"][15] = 10**1000
        self.assertIsNone(consumption_kwh(self.samples, 2, "today", self.now))

    def test_sum_overflow_in_yesterday_or_today_returns_unknown(self):
        self.samples[0]["v"][5] = 1e308
        self.samples[0]["v"][6] = 1e308
        self.samples[1]["v"][-1] = 1e308
        self.assertIsNone(consumption_kwh(self.samples, 2, "today", self.now))

        self.samples[0]["v"][5] = 2.4
        self.samples[0]["v"][6] = 0.0
        self.samples[1]["v"][-1] = 2.4
        self.samples[0]["v"][15] = 1e308
        self.samples[0]["v"][17] = 1e308
        self.assertIsNone(consumption_kwh(self.samples, 2, "today", self.now))

    def test_sum_overflow_in_month_or_year_returns_unknown(self):
        self.samples[1]["v"] = [1e308] * 19 + [2.4]
        self.assertIsNone(consumption_kwh(self.samples, 2, "current_month", self.now))

        self.samples[1]["v"] = [1.0] * 19 + [2.4]
        self.samples[2]["v"] = [1e308] * 8 + [31.5]
        self.assertIsNone(consumption_kwh(self.samples, 2, "current_year", self.now))

    def test_malformed_cloud_record_returns_unknown(self):
        self.assertIsNone(
            consumption_kwh([None, *self.samples], 2, "today", self.now)
        )

    def test_last_minute_before_month_rollover(self):
        buckets = [0.0] * 24
        buckets[4] = 2.4  # last completed day: October 30
        buckets[18] = 1.0  # current day: October 31
        sequences = [
            {"k": 2, "p": 1, "v": buckets},
            {"k": 2, "p": 3, "v": [2.4]},
        ]
        self.assertEqual(
            consumption_kwh(sequences, 2, "today", datetime(2026, 10, 31, 23, 59)),
            1.0,
        )

    def test_midnight_month_rollover_with_fresh_reporting_window(self):
        buckets = [0.0] * 24
        buckets[16] = 1.0  # October 31, now the completed previous day
        buckets[23] = 0.2  # November 1
        sequences = [
            {"k": 2, "p": 1, "v": buckets},
            {"k": 2, "p": 3, "v": [1.0]},
            {"k": 2, "p": 4, "v": [5.0] * 9 + [43.2]},
        ]
        for moment in (
            datetime(2026, 11, 1, 0, 0),
            datetime(2026, 11, 1, 0, 0, 1),
        ):
            with self.subTest(moment=moment):
                self.assertEqual(
                    consumption_kwh(sequences, 2, "today", moment), 0.2
                )
                self.assertEqual(
                    consumption_kwh(sequences, 2, "current_month", moment), 0.2
                )
                self.assertEqual(
                    consumption_kwh(sequences, 2, "previous_month", moment), 43.2
                )
                self.assertEqual(
                    consumption_kwh(sequences, 2, "current_year", moment), 88.4
                )

    def test_stale_reporting_window_at_midnight_is_rejected(self):
        buckets = [0.0] * 24
        buckets[4] = 2.4  # old previous day
        buckets[18] = 1.0  # old current day
        sequences = [
            {"k": 2, "p": 1, "v": buckets},
            {"k": 2, "p": 3, "v": [2.4]},
        ]
        self.assertIsNone(
            consumption_kwh(sequences, 2, "today", datetime(2026, 11, 1, 0, 0))
        )

    def test_api_daily_alignment_without_provisional_bucket(self):
        """Actual next-day pattern: prior-day total matches one slot earlier."""
        buckets = [
            1.85, 0.2, 0.0, 0.0, 0.175, 0.0, 0.2, 0.175,
            0.0, 0.0, 0.175, 0.325, 0.175, 0.175, 0.0, 0.125,
            0.175, 2.0, 0.2, 0.0, 0.0, 0.175, 0.0, 1.975,
        ]
        sequences = [
            {"k": 2, "p": 1, "v": buckets},
            {"k": 2, "p": 3, "v": [3.525]},
        ]
        self.assertEqual(
            consumption_kwh(sequences, 2, "today", datetime(2026, 10, 22, 11, 10)),
            2.35,
        )

    def test_different_valid_alignments_with_different_totals_fail_closed(self):
        buckets = [0.0] * 24
        buckets[5] = 2.0
        buckets[17] = 2.0
        buckets[23] = 0.5
        sequences = [
            {"k": 2, "p": 1, "v": buckets},
            {"k": 2, "p": 3, "v": [2.0]},
        ]
        # At 11:10, yesterday may end at slot 17 or 18. Both can
        # appear consistent, but the candidate 'today' totals differ.
        self.assertIsNone(
            consumption_kwh(sequences, 2, "today", datetime(2026, 10, 22, 11, 10))
        )

    def test_unknown_period_raises(self):
        with self.assertRaises(ValueError):
            consumption_kwh(self.samples, 2, "lifetime", self.now)
