"""Check Home Assistant's entity translation schema for DHW period sensors."""

import json
from pathlib import Path
from unittest import TestCase

INTEGRATION = Path(__file__).parent.parent / "custom_components" / "ariston_net"
PERIOD_KEYS = {
    "dhw_energy_today",
    "dhw_energy_current_month",
    "dhw_energy_previous_month",
    "dhw_energy_current_year",
}
LANGUAGES = ("ca", "en", "es", "fr", "it", "pl", "pt", "ru", "ua")


class EnergyTranslationTests(TestCase):
    def test_all_supported_languages_have_entity_sensor_names(self):
        for path in (INTEGRATION / "strings.json",) + tuple(
            INTEGRATION / "translations" / f"{lang}.json" for lang in LANGUAGES
        ):
            with self.subTest(file=path.name):
                data = json.loads(path.read_text(encoding="utf-8"))
                self.assertNotIn("sensor", data)
                self.assertEqual(set(data["entity"]["sensor"]), PERIOD_KEYS)
                for entity in data["entity"]["sensor"].values():
                    self.assertTrue(entity["name"].strip())

    def test_locale_files_use_readable_utf8(self):
        for lang in LANGUAGES:
            with self.subTest(lang=lang):
                content = (
                    INTEGRATION / "translations" / f"{lang}.json"
                ).read_text(encoding="utf-8")
                self.assertNotIn(r"\u", content)
