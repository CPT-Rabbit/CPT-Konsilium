from __future__ import annotations

import unittest

from konsilium.declutter import declutter, declutter_stats


class DeclutterTest(unittest.TestCase):
    def test_strips_institutional_boilerplate_keeps_clinical(self) -> None:
        text = (
            "Diagnose: Selbstlimitierende fokale Epilepsie\n"
            "Anamnese: taeglich mehrfach Ereignisse mit Kopfreklination\n"
            "IBAN DE00 1234 5678 9012 3456 78\n"
            "BIC MUSTDEHHXXX\n"
            "Amtsgericht Hamburg HRB 12 345\n"
            "Seite 2 von 4\n"
            "www.kkh-musterstift.de\n"
            "Empfehlung: keine anfallssupprimierende Medikation\n"
        )
        out = declutter(text)
        self.assertIn("Selbstlimitierende fokale Epilepsie", out)
        self.assertIn("Kopfreklination", out)
        self.assertIn("keine anfallssupprimierende Medikation", out)
        for noise in ("IBAN", "BIC", "Amtsgericht", "HRB", "Seite 2 von 4", "www."):
            self.assertNotIn(noise, out)

    def test_stats_categorize_removed_lines(self) -> None:
        text = "Klinischer Befund unauffaellig\nIBAN DE00\nSeite 1/3\nTelefon: 040 12345\n"
        stats = declutter_stats(text)
        self.assertEqual(stats.get("reqs"), 1)
        self.assertEqual(stats.get("page"), 1)
        self.assertEqual(stats.get("contact"), 1)

    def test_never_drops_clinical_line_with_numbers(self) -> None:
        # lab/vitals lines carry digits but no institutional markers — must stay
        text = "Gewicht 40 kg, Laenge 143 cm, BMI 19,6 kg/m2\nEEG: sharp-wave-Fokus rechts C4\n"
        self.assertEqual(declutter(text).strip(), text.strip())


if __name__ == "__main__":
    unittest.main()
