"""Contract tests against public Ariston API responses in upstream issue #55.

These snapshots have no acquisition timestamps. We deliberately do not assert
calendar-day/month-to-date totals using arbitrary dates.
"""

import json
from pathlib import Path
from unittest import TestCase

from custom_components.ariston_net.energy_series import (
    _consumption_samples,
    consumption_kwh,
)

FIXTURE_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "issue_55_consumption_sequences.json"
)


class Issue55EnergyPayloadTests(TestCase):
    @classmethod
    def setUpClass(cls):
        with FIXTURE_PATH.open(encoding="utf-8") as file:
            cls.devices = json.load(file)["devices"]

    def test_all_four_upstream_models_and_sequences_are_preserved(self):
        self.assertEqual(
            {name: len(device["sequences"]) for name, device in self.devices.items()},
            {
                "lydos_hybrid": 10,
                "velis_evo": 4,
                "alteas_one_24": 8,
                "genus_one": 24,
            },
        )

    def test_all_reported_types_and_periods_are_selectable(self):
        """No device-specific hardcoding or positional indexing in the parser."""
        for model, device in self.devices.items():
            sequences = device["sequences"]
            for sequence in sequences:
                with self.subTest(model=model, k=sequence["k"], p=sequence["p"]):
                    samples = _consumption_samples(
                        sequences, sequence["k"], sequence["p"]
                    )
                    self.assertEqual(samples, sequence["v"])
                    self.assertTrue(all(isinstance(v, float) for v in samples))

    def test_last_week_matches_last_seven_daily_samples_with_rounding(self):
        """Most upstream examples agree exactly; Alteas k=2 differs by 0.002."""
        for model, device in self.devices.items():
            sequences = device["sequences"]
            kinds = {seq["k"] for seq in sequences if seq["p"] == 2}
            for kind in kinds:
                weekly = _consumption_samples(sequences, kind, 2)
                daily = _consumption_samples(sequences, kind, 3)
                if daily is None:
                    continue
                with self.subTest(model=model, k=kind):
                    self.assertEqual(len(weekly), 7)
                    self.assertGreaterEqual(len(daily), 7)
                    for actual, expected in zip(weekly, daily[-7:]):
                        self.assertAlmostEqual(actual, expected, delta=0.0021)

    def test_reported_subdaily_bucket_shapes_differ_across_models(self):
        """The API never supplies explicit timestamps or bucket lengths."""
        for model, expected_length in {
            "lydos_hybrid": 24,
            "velis_evo": 24,
            "alteas_one_24": 12,
            "genus_one": 24,
        }.items():
            sequences = self.devices[model]["sequences"]
            for sequence in sequences:
                if sequence["p"] != 1:
                    continue
                with self.subTest(model=model, k=sequence["k"]):
                    self.assertEqual(len(sequence["v"]), expected_length)

    def test_alteas_twelve_bucket_profile_is_not_misread_as_48_hours(self):
        """Do not turn an unsupported 12-bucket snapshot into a daily total."""
        from datetime import datetime

        sequences = self.devices["alteas_one_24"]["sequences"]
        self.assertIsNone(
            consumption_kwh(
                sequences, 2, "today", datetime(2026, 10, 21, 16, 30),
                bucket_minutes=120,
            )
        )

    def test_last_bucket_is_not_same_as_latest_completed_day(self):
        """This was the original v[-1] bug: the last p=1 entry is not a day."""
        series = self.devices["velis_evo"]["sequences"]
        latest_bucket = _consumption_samples(series, 2, 1)[-1]
        latest_completed_day = _consumption_samples(series, 2, 2)[-1]
        self.assertEqual(latest_bucket, 2.025)
        self.assertEqual(latest_completed_day, 7.65)
        self.assertNotEqual(latest_bucket, latest_completed_day)
