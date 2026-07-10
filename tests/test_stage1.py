from __future__ import annotations

import json
import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from konsilium import (
    EgressViolation,
    ResidueError,
    assert_safe_knowledge_query,
    deidentify,
    deid_preview,
    ingest_document,
    ingest_from_preview,
    ingest_patient_file,
    ingest_patient_document,
    ingest_text,
    residue_report,
    stage1_smoke,
)
from konsilium.config import Config
from konsilium.deid import PiiEntity
from konsilium.ingest import _ollama_detector, extract_text_with_stats
from konsilium.memory import PatientMemory
from konsilium.ollama_deid import OllamaPiiDetector


class Stage1Test(unittest.TestCase):
    def test_ingest_keeps_pii_out_of_patient_memory(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            document_path = ingest_text(
                "case-1",
                "\n".join(
                    [
                        "Patient: Anna Mueller",
                        "Geburtsdatum: 12.03.1974",
                        "Adresse: Hauptstrasse 7, Berlin",
                        "KVNR: X123456789",
                        "Visit date: 2026-07-01",
                        "HbA1c elevated, metformin discussed.",
                    ]
                ),
                root,
                allow_synthetic=True,
            )

            patient_text = document_path.read_text(encoding="utf-8")
            self.assertIn("[PATIENT_1]", patient_text)
            self.assertRegex(patient_text, r"age \d+")
            self.assertNotIn("[DOB_1]", patient_text)
            self.assertIn("Visit date: 2026-07-01", patient_text)
            self.assertNotIn("Anna Mueller", patient_text)
            self.assertNotIn("12.03.1974", patient_text)
            self.assertNotIn("Hauptstrasse", patient_text)

            vault = json.loads((root / "identity_vault" / "case-1.json").read_text(encoding="utf-8"))
            self.assertEqual(vault["[PATIENT_1]"], "Anna Mueller")
            self.assertEqual(vault["[DOB_1]"], "12.03.1974")

    def test_regex_deid_catches_german_insurance_numbers_without_model(self) -> None:
        document = deidentify(
            "\n".join(
                [
                    "Versichertennummer A987654321",
                    "Die Karte A987654321 wurde vorgelegt.",
                    "HbA1c 8.1, LDL 140, CRP 5",
                ]
            )
        )

        self.assertIn("[INSURANCE_1]", document.text)
        self.assertEqual(document.text.count("[INSURANCE_1]"), 2)
        self.assertNotIn("A987654321", document.text)
        self.assertIn("HbA1c 8.1", document.text)
        self.assertIn("LDL 140", document.text)
        self.assertEqual(document.vault["[INSURANCE_1]"], "A987654321")

    def test_regex_deid_handles_realistic_german_letterhead_patterns(self) -> None:
        document = deidentify(
            "\n".join(
                [
                    "EEG Befund",
                    "Musterstraße 12",
                    "Muster strasse 13",
                    "10115 Berlin",
                    "Fallnummer 12345678",
                    "geb . 04 . 05 . 1962",
                    "Tel.: +49 (0)30 / 123-4567",
                    "Fax: 030-7654321",
                ]
            )
        )

        for value in ("Musterstraße 12", "Muster strasse 13", "10115 Berlin", "12345678", "04 . 05 . 1962", "123-4567", "7654321"):
            self.assertNotIn(value, document.text)
        self.assertIn("[CASE_NUMBER_1]", document.text)
        self.assertRegex(document.text, r"age \d+")
        self.assertFalse(
            {hit.pattern for hit in residue_report(document.text)}
            & {"DOB", "STREET", "PLZ_CITY", "CASE_NUMBER", "PHONE"}
        )

    def test_institutional_letterhead_stays_while_physician_name_is_tokenized(self) -> None:
        document = deidentify(
            "\n".join(
                [
                    "Klinikum Musterstadt",
                    "Dr. med. Holger Steinbrecher",
                    "Musterstraße 12",
                    "10115 Berlin",
                    "Tel.: 030 1234567",
                    "Fax: 030 7654321",
                    "www.klinikum.example",
                ]
            ),
            pii_detector=lambda text: [
                PiiEntity("PERSON", "Dr. med. Holger Steinbrecher"),
                PiiEntity("ADDRESS", "Musterstraße 12"),
                PiiEntity("PHONE", "030 1234567"),
            ],
        )

        self.assertIn("Musterstraße 12", document.text)
        self.assertIn("10115 Berlin", document.text)
        self.assertIn("030 1234567", document.text)
        self.assertNotIn("Holger Steinbrecher", document.text)
        self.assertIn("[PATIENT_1]", document.text)
        self.assertFalse(
            {hit.pattern for hit in residue_report(document.text)}
            & {"STREET", "PLZ_CITY", "PHONE", "DIGIT_RUN"}
        )

    def test_recipient_address_adjacent_to_person_is_private(self) -> None:
        source = "\n".join([
            "Klinikum Musterstadt",
            "Frau",
            "Erika Beispiel",
            "Striepenweg 41",
            "21147 Hamburg",
        ])
        document = deidentify(
            source,
            pii_detector=lambda text: [PiiEntity("PERSON", "Erika Beispiel")],
        )

        self.assertNotIn("Erika Beispiel", document.text)
        self.assertNotIn("Striepenweg 41", document.text)
        self.assertNotIn("21147 Hamburg", document.text)
        self.assertRegex(document.text, r"\[ADDR_\d+\]")

        leaked_preview = source.replace("Erika Beispiel", "[PATIENT_13]")
        patterns = {hit.pattern for hit in residue_report(leaked_preview)}
        self.assertIn("STREET", patterns)
        self.assertIn("PLZ_CITY", patterns)

    def test_model_address_entities_reject_clinical_words_consistently(self) -> None:
        source = "\n".join([
            "Es seien keine bilateral tonisch-klonische Anfälle beobachtet worden.",
            "Keine Auffälligkeiten und keine expressive Sprache.",
            "Dies erfolgte ohne Sedierung und nicht stationär.",
            "Hinweise zu seiner zweisprachigen Erziehung.",
            "Sehr geehrte Frau Kollegin,",
        ])
        document = deidentify(
            source,
            pii_detector=lambda text: [
                PiiEntity("ADDRESS", "keine"),
                PiiEntity("ADDRESS", "zweisprachigen"),
                PiiEntity("ADDRESS", "Sehr"),
                PiiEntity("ADDRESS", "Frau Kollegin"),
            ],
        )

        self.assertEqual(document.text, source)
        self.assertNotIn("[ADDR_", document.text)
        self.assertEqual(
            [(item.kind, item.reason) for item in document.rejected_entities],
            [
                ("ADDR", "generic_word"),
                ("ADDR", "not_address_like"),
                ("ADDR", "generic_word"),
                ("ADDR", "not_address_like"),
            ],
        )

    def test_institutional_city_is_retained_but_patient_residence_is_tokenized(self) -> None:
        institutional = "Hamburg, den 10.09.2025\nAmtsgericht Hamburg"
        institution_document = deidentify(
            institutional,
            pii_detector=lambda text: [PiiEntity("ADDRESS", "Hamburg")],
        )
        residence_document = deidentify(
            "[PATIENT_1] lebt auf Rügen.",
            pii_detector=lambda text: [PiiEntity("ADDRESS", "Rügen")],
        )

        self.assertEqual(institution_document.text, institutional)
        self.assertNotIn("Rügen", residence_document.text)
        self.assertIn("[ADDR_1]", residence_document.text)

    def test_ambiguous_bare_address_is_tokenized(self) -> None:
        document = deidentify("Musterstraße 12\n10115 Berlin")

        self.assertNotIn("Musterstraße 12", document.text)
        self.assertNotIn("10115 Berlin", document.text)

    def test_entity_replacement_preserves_words_and_existing_tokens(self) -> None:
        document = deidentify(
            "Rolandofoki Roland B [DOB_2]",
            pii_detector=lambda text: [
                PiiEntity("PERSON", "B"),
                PiiEntity("PERSON", "Roland"),
                PiiEntity("PERSON", "DOB"),
            ],
        )

        self.assertEqual(document.text, "Rolandofoki [PATIENT_1] B [DOB_2]")
        self.assertEqual(document.vault, {"[PATIENT_1]": "Roland"})

    def test_longest_person_entity_replaces_before_shorter_overlap(self) -> None:
        document = deidentify(
            "Anna Müller",
            pii_detector=lambda text: [PiiEntity("PERSON", "Anna"), PiiEntity("PERSON", "Anna Müller")],
        )

        self.assertEqual(document.text, "[PATIENT_1]")
        self.assertEqual(document.vault, {"[PATIENT_1]": "Anna Müller"})

    def test_residue_gate_blocks_corrupted_token_syntax(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ResidueError, r"CORRUPTED_TOKEN lines 1"):
                ingest_text("case-1", "[DO[PATIENT_8]_2]", root, allow_synthetic=True)

            self.assertFalse((root / "patients").exists())

        patterns = {hit.pattern for hit in residue_report("[PATIENT_1")}
        self.assertIn("CORRUPTED_TOKEN", patterns)

    def test_regex_deid_handles_page_headers_spelled_dob_and_spaced_case_numbers(self) -> None:
        self.assertIn("PERSON_HEADER", {hit.pattern for hit in residue_report("Seite 1 von 2, Mueller,")})
        document = deidentify(
            "\n".join(
                [
                    "Patienten Mueller,",
                    "Seite 1 von 2, Mueller,",
                    "Geburtsdatum Sonntag, 19. Juli 2015",
                    "geb. 19.07.201 5",
                    "wohnhaft: Feldstieg 7",
                    "Musterstr. 8",
                    "Pat. Nr. 12345678",
                    "Aufn. Nr. A1234567",
                ]
            )
        )

        for value in ("Mueller", "Sonntag, 19. Juli 2015", "19.07.201 5", "Feldstieg 7", "Musterstr. 8", "12345678", "A1234567"):
            self.assertNotIn(value, document.text)
        self.assertEqual(document.text.count("[PATIENT_1]"), 2)
        self.assertGreaterEqual(document.text.count("age "), 2)
        self.assertFalse(
            {hit.pattern for hit in residue_report(document.text)}
            & {"DOB", "PERSON_HEADER", "STREET", "CASE_NUMBER"}
        )

    def test_footer_case_number_after_split_year_dob_is_tokenized_before_detector(self) -> None:
        header = "Seite 2 von 4, Mustermann, Erika geb. am 19.07.201 5, 123456"

        def detector(text):
            self.assertEqual(
                text,
                "Seite 2 von 4, [PATIENT_1], Erika geb. am age 10, [CASE_NUMBER_1]",
            )
            return [
                PiiEntity("DOB", "10, 123456"),
                PiiEntity("PERSON", "Erika"),
            ]

        document = deidentify(
            header,
            pii_detector=detector,
            today=date(2026, 7, 10),
        )

        self.assertIn("[PATIENT_1], [PATIENT_2] geb. am age 10, [CASE_NUMBER_1]", document.text)
        self.assertNotIn("age [DOB_", document.text)
        self.assertNotIn("[DOB_", document.text)
        self.assertIn("19.07.201 5", document.vault.values())
        self.assertEqual(residue_report(document.text), [])

    def test_ocr_month_dob_is_age_converted_with_only_partial_name_residue(self) -> None:
        source = "09.09.2025 21:29:03 [PATIENT_1] larosiav, Geburtsdatum Sonntag, 19. JuËI 2015"
        document = deidentify(source, today=date(2026, 7, 10))

        self.assertIn("Geburtsdatum age 10", document.text)
        self.assertNotIn("Sonntag, 19. JuËI 2015", document.text)
        self.assertNotIn("[DOB_", document.text)
        self.assertIn("Sonntag, 19. JuËI 2015", document.vault.values())
        self.assertEqual({hit.pattern for hit in residue_report(document.text)}, {"PARTIAL_NAME"})

    def test_detector_sourced_dob_uses_age_conversion(self) -> None:
        document = deidentify(
            "Geburtsdatum ist 19.07.2015",
            pii_detector=lambda text: [PiiEntity("DOB", "19.07.2015")],
            today=date(2026, 7, 10),
        )

        self.assertEqual(document.text, "Geburtsdatum ist age 10")
        self.assertNotIn("[DOB_", document.text)
        self.assertIn("19.07.2015", document.vault.values())
        self.assertEqual(residue_report(document.text), [])

    def test_compact_physician_names_are_tokenized(self) -> None:
        document = deidentify("Dr. B.Püst\nDr. B.Kohl\nFrau Dr. Preuße")

        for value in ("Dr. B.Püst", "Dr. B.Kohl", "Frau Dr. Preuße"):
            self.assertNotIn(value, document.text)
        self.assertEqual(document.text.count("[PATIENT_"), 3)

    def test_ocr_email_gate_and_institutional_email_decision(self) -> None:
        private = deidentify("kontakt patient@example . de")
        self.assertNotIn("patient@example . de", private.text)

        institutional = deidentify("Klinik Musterstadt\ninfo@klinikum . de")
        self.assertIn("info@klinikum . de", institutional.text)
        self.assertEqual(institutional.retained_institutional_emails, ("info@klinikum . de",))
        self.assertIn("EMAIL", {hit.pattern for hit in residue_report(institutional.text)})
        self.assertNotIn(
            "EMAIL",
            {
                hit.pattern
                for hit in residue_report(
                    institutional.text,
                    retained_institutional_emails=institutional.retained_institutional_emails,
                )
            },
        )
        self.assertIn("EMAIL", {hit.pattern for hit in residue_report("raw@mail.de")})

        named = deidentify("Klinik Musterstadt\n.pferner@aktion-sennenschein-greifswald de")
        self.assertNotIn(".pferner@aktion-sennenschein-greifswald de", named.text)
        self.assertEqual(named.retained_institutional_emails, ())

    def test_partial_name_next_to_patient_token_on_dob_line_blocks(self) -> None:
        leaked = "09.09.2025 18:18:31 [PATIENT_1] 3arosiav, Geburtsdatum age 10"
        tokenized_header = "Seite 2 von 4, [PATIENT_1], [PATIENT_12] geb. am age 10"

        self.assertIn("PARTIAL_NAME", {hit.pattern for hit in residue_report(leaked)})
        self.assertNotIn("PARTIAL_NAME", {hit.pattern for hit in residue_report(tokenized_header)})
        self.assertNotIn(
            "PARTIAL_NAME",
            {hit.pattern for hit in residue_report("[PATIENT_5] wurde heute untersucht")},
        )

    def test_ocr_spaced_private_plz_is_tokenized_and_gated(self) -> None:
        source = "nachrichtlich: [PATIENT_3], [ADDR_2], 18 528 Bergen"
        document = deidentify(source)

        self.assertNotIn("18 528 Bergen", document.text)
        self.assertIn("PLZ_CITY", {hit.pattern for hit in residue_report(source)})

    def test_reviewed_preview_ingest_requires_clean_gate_and_uses_preview_vault(self) -> None:
        def structure_model(messages, system_prompt):
            return {"timeline": [], "problems": ["Kontrolltermin"], "meds": [], "labs": []}

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            previews = root / "previews"
            previews.mkdir()
            preview = previews / "preview-ocr.md"
            preview.write_text(
                "09.09.2025 18:18:31 [PATIENT_1] 3arosiav, Geburtsdatum age 10\nProblem: Kontrolltermin",
                encoding="utf-8",
            )
            preview.with_suffix(".vault.json").write_text(
                json.dumps({"[PATIENT_1]": "Maria Beispiel"}),
                encoding="utf-8",
            )
            config = Config.model_validate({
                "runtime": {"patient_root": root, "allow_real_patient_docs": True},
            })

            with self.assertRaisesRegex(ResidueError, "PARTIAL_NAME lines 1"):
                ingest_from_preview(config, "case-1", preview, structure_model=structure_model)
            self.assertFalse((root / "patients").exists())

            preview.write_text(
                "09.09.2025 18:18:31 [PATIENT_1], Geburtsdatum age 10\nProblem: Kontrolltermin",
                encoding="utf-8",
            )
            document_path = ingest_from_preview(config, "case-1", preview, structure_model=structure_model)

            self.assertIn("[PATIENT_1]", document_path.read_text(encoding="utf-8"))
            vault = json.loads((root / "identity_vault" / "case-1.json").read_text(encoding="utf-8"))
            self.assertEqual(vault, {"[PATIENT_1]": "Maria Beispiel"})

    def test_institutional_ust_id_does_not_trigger_digit_residue(self) -> None:
        text = "Klinik Musterstadt\nUst-ID Nr. DE252426446"

        self.assertNotIn("DIGIT_RUN", {hit.pattern for hit in residue_report(text)})

    def test_residue_gate_blocks_without_writing_patient_memory(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ResidueError, r"DIGIT_RUN lines 1") as raised:
                ingest_text("case-1", "OCR noise 123456", root, allow_synthetic=True)

            self.assertNotIn("123456", str(raised.exception))
            self.assertFalse((root / "patients").exists())
            self.assertFalse((root / "identity_vault").exists())

        reported = residue_report("OCR noise 123456", {"DIGIT_RUN": "report"})
        self.assertEqual(len(reported), 1)
        self.assertEqual(reported[0].action, "report")

    def test_deid_preview_writes_only_local_preview_artifacts(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "letterhead.txt"
            source.write_text(
                "Patient: Anna Mueller\nMusterstrasse 7\n80331 Muenchen\ngeb. 04.05.1962\nFall-Nr 12345678",
                encoding="utf-8",
            )

            result = deid_preview(
                Config.model_validate({"runtime": {"patient_root": root}}),
                source,
                pii_detector=lambda text: [],
            )

            preview = Path(result["preview_path"]).read_text(encoding="utf-8")
            vault = Path(result["vault_path"]).read_text(encoding="utf-8")
            self.assertNotIn("Anna Mueller", preview)
            self.assertNotIn("Musterstrasse 7", preview)
            self.assertIn("Anna Mueller", vault)
            self.assertFalse(result["residue"]["blocked"])
            self.assertNotIn("Anna Mueller", json.dumps(result, ensure_ascii=False))
            self.assertFalse((root / "patients").exists())
            self.assertFalse((root / "identity_vault").exists())

    def test_deid_preview_reports_rejected_entities_without_values(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "clinical-note.txt"
            source.write_text("Es wurden keine Anfälle beobachtet.", encoding="utf-8")

            result = deid_preview(
                Config.model_validate({"runtime": {"patient_root": root}}),
                source,
                pii_detector=lambda text: [PiiEntity("ADDRESS", "keine")],
            )

            report = json.loads(Path(result["residue_report_path"]).read_text(encoding="utf-8"))
            self.assertEqual(report["rejected_entities"], [{"kind": "ADDR", "reason": "generic_word"}])
            self.assertNotIn("keine", json.dumps(report, ensure_ascii=False))

    def test_egress_guard_rejects_tokens_and_raw_pii(self) -> None:
        assert_safe_knowledge_query("metformin hba1c older adults guideline")

        with self.assertRaises(EgressViolation):
            assert_safe_knowledge_query("metformin for [PATIENT_1]")

        with self.assertRaises(EgressViolation):
            assert_safe_knowledge_query("diabetes Geburtsdatum: 12.03.1974")

    def test_ingest_appends_documents_and_preserves_existing_tokens(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_path = ingest_text(
                "case-1",
                "Patient: Anna Mueller\n2026-07-01 HbA1c 8.1",
                root,
                allow_synthetic=True,
            )
            second_path = ingest_text(
                "case-1",
                "Patient: Anna Mueller\n2026-07-08 LDL 140",
                root,
                allow_synthetic=True,
            )

            patient_text = first_path.read_text(encoding="utf-8") + second_path.read_text(encoding="utf-8")
            self.assertIn("2026-07-01 HbA1c 8.1", patient_text)
            self.assertIn("2026-07-08 LDL 140", patient_text)
            self.assertEqual(patient_text.count("[PATIENT_1]"), 2)
            self.assertEqual(second_path.stem, first_path.stem + "-2")

            vault = json.loads((root / "identity_vault" / "case-1.json").read_text())
            self.assertEqual(vault, {"[PATIENT_1]": "Anna Mueller"})

    def test_ingest_names_document_from_structured_metadata_and_indexes_it(self) -> None:
        def structure_model(messages, system_prompt):
            self.assertIn("document_topic", system_prompt)
            return {
                "timeline": [],
                "problems": [],
                "meds": [],
                "labs": [],
                "document_date": "2025-09-10",
                "document_topic": "EEG Befund",
                "document_sender": "Wilhelmstift",
            }

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            document_path = ingest_text(
                "case-1",
                "Hamburg, den 10.09.2025\nEEG ohne epilepsietypische Potentiale.",
                root,
                structure_model=structure_model,
                allow_synthetic=True,
            )

            self.assertEqual(document_path.name, "2025-09-10_EEG-Befund_Wilhelmstift.md")
            self.assertIn("date_source: document", document_path.read_text(encoding="utf-8"))
            hits = PatientMemory(root).search("EEG Befund Wilhelmstift", patient_id="case-1")
            self.assertIn(str(document_path.relative_to(root)), {hit["path"] for hit in hits})

    def test_document_filename_fallbacks_are_safe_and_mark_ingest_date(self) -> None:
        responses = iter([
            {
                "document_date": "",
                "document_topic": "Ärztlicher Bericht",
                "document_sender": "Klinikum München",
            },
            {
                "document_date": "2025-09-10",
                "document_topic": "[PATIENT_1]",
                "document_sender": "Fallnummer 123456",
            },
        ])

        def structure_model(messages, system_prompt):
            return {"timeline": [], "problems": [], "meds": [], "labs": [], **next(responses)}

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            fallback_path = ingest_text(
                "case-1",
                "Befund ohne Briefdatum.",
                root,
                structure_model=structure_model,
                allow_synthetic=True,
            )
            unsafe_path = ingest_text(
                "case-1",
                "Hamburg, den 10.09.2025\nKontrollbefund.",
                root,
                structure_model=structure_model,
                allow_synthetic=True,
            )

            self.assertEqual(
                fallback_path.name,
                f"{date.today().isoformat()}_Aerztlicher-Bericht_Klinikum-Muenchen.md",
            )
            self.assertIn("date_source: ingest", fallback_path.read_text(encoding="utf-8"))
            self.assertEqual(unsafe_path.name, "2025-09-10_document_unknown-sender.md")
            self.assertNotIn("PATIENT", unsafe_path.name)
            self.assertNotIn("123456", unsafe_path.name)
            self.assertEqual(residue_report(unsafe_path.name), [])

    def test_ingest_document_uses_pdf_extractor_hook(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf = root / "synthetic.pdf"
            pdf.write_bytes(b"%PDF synthetic fixture")

            document_path = ingest_document(
                "case-1",
                pdf,
                root,
                extractor=lambda path: "Patient: Anna Mueller\n2026-07-01 HbA1c 8.1",
                allow_synthetic=True,
            )

            patient_text = document_path.read_text(encoding="utf-8")
            self.assertIn("[PATIENT_1]", patient_text)
            self.assertNotIn("Anna Mueller", patient_text)

    def test_pdf_extraction_tracks_text_layer_ocr_and_mixed_pages(self) -> None:
        class Page:
            def __init__(self, text: str):
                self.text = text

            def extract_text(self) -> str:
                return self.text

        def reader(path: Path):
            if path.name == "text.pdf":
                return [Page("Patient: Anna Mueller\nProblem: Diabetes")]
            if path.name == "scan.pdf":
                return [Page("")]
            return [Page("Patient: Anna Mueller"), Page("")]

        def ocr(path: Path, pages: list[int]) -> dict[int, str]:
            return {page: f"OCR page {page}: Anna Mueller HbA1c 8.1" for page in pages}

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("text.pdf", "scan.pdf", "mixed.pdf"):
                (root / name).write_bytes(b"%PDF fixture")

            text = extract_text_with_stats(root / "text.pdf", pdf_reader=reader, ocr_pages=ocr)
            scan = extract_text_with_stats(root / "scan.pdf", pdf_reader=reader, ocr_pages=ocr)
            mixed = extract_text_with_stats(root / "mixed.pdf", pdf_reader=reader, ocr_pages=ocr)

            self.assertEqual(text.stats["text_layer_pages"], [1])
            self.assertEqual(text.stats["ocr_pages"], [])
            self.assertEqual(scan.stats["text_layer_pages"], [])
            self.assertEqual(scan.stats["ocr_pages"], [1])
            self.assertEqual(mixed.stats["text_layer_pages"], [1])
            self.assertEqual(mixed.stats["ocr_pages"], [2])
            self.assertIn("OCR page 2", mixed.text)

    def test_pdf_ocr_output_is_deidentified_and_empty_pages_fail_loudly(self) -> None:
        class Page:
            def extract_text(self) -> str:
                return ""

        def reader(path: Path):
            return [Page()]

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf = root / "scan.pdf"
            pdf.write_bytes(b"%PDF fixture")

            def ocr(path: Path, pages: list[int]) -> dict[int, str]:
                return {1: "Patient: Anna Mueller\nProblem: Diabetes"}

            document_path = ingest_document(
                "case-1",
                pdf,
                root,
                extractor=lambda path: extract_text_with_stats(path, pdf_reader=reader, ocr_pages=ocr).text,
                allow_synthetic=True,
            )
            patient_text = document_path.read_text(encoding="utf-8")
            self.assertIn("[PATIENT_1]", patient_text)
            self.assertNotIn("Anna Mueller", patient_text)

            with self.assertRaisesRegex(RuntimeError, "pages without text after OCR: 1"):
                extract_text_with_stats(pdf, pdf_reader=reader, ocr_pages=lambda path, pages: {1: ""})

            with self.assertRaisesRegex(RuntimeError, "PDF has no pages"):
                extract_text_with_stats(pdf, pdf_reader=lambda path: [])

            def corrupt_reader(path: Path):
                raise RuntimeError("corrupt PDF")

            with self.assertRaisesRegex(RuntimeError, "corrupt PDF"):
                extract_text_with_stats(pdf, pdf_reader=corrupt_reader)

            with patch("konsilium.ingest.shutil.which", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "OCR required but not available"):
                    extract_text_with_stats(pdf, pdf_reader=reader)

    def test_local_detector_removes_free_text_german_and_english_pii(self) -> None:
        def detector(text: str) -> list[PiiEntity]:
            return [
                PiiEntity("PATIENT", "Frau Mueller"),
                PiiEntity("ADDR", "Hauptstrasse 7"),
                PiiEntity("PATIENT", "John Smith"),
                PiiEntity("ADDR", "742 Evergreen Terrace"),
            ]

        document = deidentify(
            "\n".join(
                [
                    "Frau Mueller berichtet über Kopfschmerzen in Hauptstrasse 7.",
                    "John Smith reports nausea at 742 Evergreen Terrace.",
                ]
            ),
            pii_detector=detector,
        )

        self.assertIn("[PATIENT_1]", document.text)
        self.assertIn("[ADDR_1]", document.text)
        self.assertIn("[PATIENT_2]", document.text)
        self.assertRegex(document.text, r"\[ADDR_\d+\]")
        self.assertNotIn("Frau Mueller", document.text)
        self.assertNotIn("Hauptstrasse 7", document.text)
        self.assertNotIn("John Smith", document.text)
        self.assertNotIn("742 Evergreen Terrace", document.text)

    def test_ingest_passes_local_detector_into_deid_pipeline(self) -> None:
        def detector(text: str) -> list[PiiEntity]:
            return [PiiEntity("PATIENT", "Frau Mueller")]

        def structure_model(messages, system_prompt):
            return {"timeline": [], "problems": ["Kopfschmerzen"], "meds": [], "labs": []}

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            document_path = ingest_text(
                "case-1",
                "Frau Mueller berichtet über Kopfschmerzen.",
                root,
                pii_detector=detector,
                structure_model=structure_model,
            )

            patient_text = document_path.read_text(encoding="utf-8")
            self.assertIn("[PATIENT_1]", patient_text)
            self.assertNotIn("Frau Mueller", patient_text)

    def test_ingest_without_detector_is_not_implicit_regex_only(self) -> None:
        with TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                ingest_text("case-1", "Patient: Anna Mueller", Path(tmp))
            with self.assertRaises(RuntimeError):
                ingest_text("case-1", "Patient: Anna Mueller", Path(tmp), pii_detector=lambda text: [])

    def test_structuring_model_extracts_deidentified_german_and_english_text(self) -> None:
        class FakeStructureModel:
            def __init__(self) -> None:
                self.calls = []

            def build_kwargs(self, messages, system_prompt, tools, *, json_mode=False):
                call = {
                    "messages": messages,
                    "system_prompt": system_prompt,
                    "stream": not json_mode,
                    "max_tokens": 4096,
                }
                if json_mode:
                    call["response_format"] = {"type": "json_object"}
                self.calls.append(call)
                return call

            def call(self, kwargs):
                return SimpleNamespace(content=json.dumps({
                    "timeline": ["2026-07-02 nephrology review"],
                    "problems": ["chronische Nierenkrankheit", "iron deficiency anemia"],
                    "meds": ["Apixaban 5 mg zweimal taeglich", "rivaroxaban 20 mg nightly"],
                    "labs": ["Kreatinin 1.8 mg/dl", "ferritin 12 ng/mL"],
                }))

        def detector(text: str) -> list[PiiEntity]:
            return [
                PiiEntity("PATIENT", "Frau Mueller"),
                PiiEntity("PATIENT", "John Smith"),
                PiiEntity("ADDR", "Hauptstrasse 7"),
            ]

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = FakeStructureModel()
            document_path = ingest_text(
                "case-1",
                "\n".join(
                    [
                        "Patient: Anna Mueller",
                        "Frau Mueller wohnt in Hauptstrasse 7.",
                        "2026-07-02 Kreatinin 1.8 mg/dl, Apixaban 5 mg zweimal taeglich.",
                        "John Smith reports ferritin 12 ng/mL and rivaroxaban 20 mg nightly.",
                    ]
                ),
                root,
                pii_detector=detector,
                structure_model=model,
            )

            patient_dir = document_path.parents[1]
            labs = (patient_dir / "labs" / "labs.md").read_text(encoding="utf-8")
            meds = (patient_dir / "meds.md").read_text(encoding="utf-8")
            problems = (patient_dir / "problems.md").read_text(encoding="utf-8")
            prompt_payload = json.dumps(model.calls, ensure_ascii=False)

            self.assertIn("Kreatinin 1.8 mg/dl", labs)
            self.assertIn("ferritin 12 ng/mL", labs)
            self.assertIn("Apixaban 5 mg", meds)
            self.assertIn("rivaroxaban 20 mg", meds)
            self.assertIn("chronische Nierenkrankheit", problems)
            self.assertIn("iron deficiency anemia", problems)
            self.assertNotIn("Anna Mueller", prompt_payload)
            self.assertNotIn("Frau Mueller", prompt_payload)
            self.assertNotIn("John Smith", prompt_payload)
            self.assertNotIn("Hauptstrasse", prompt_payload)
            self.assertFalse(model.calls[0]["stream"])
            self.assertEqual(model.calls[0]["response_format"], {"type": "json_object"})

    def test_runtime_ingest_requires_real_docs_enabled_model_and_reachable_detector(self) -> None:
        config = Config()
        with TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                ingest_patient_document(config, "case-1", "Patient: Anna Mueller", Path(tmp))

            def extractor(path: Path) -> str:
                raise AssertionError("real file extractor must not run before runtime gate")

            blocked_pdf = Path(tmp) / "blocked.pdf"
            blocked_pdf.write_bytes(b"%PDF fixture")
            with self.assertRaises(RuntimeError):
                ingest_patient_file(config, "case-1", blocked_pdf, Path(tmp), extractor=extractor)

        config = Config.model_validate(
            {
                "runtime": {"allow_real_patient_docs": True},
                "deidentification": {"ollama_model": "local-med-ner"},
            }
        )

        def unreachable(text: str) -> list[PiiEntity]:
            raise RuntimeError("ollama unavailable")

        with TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                ingest_patient_document(
                    config,
                    "case-1",
                    "Patient: Anna Mueller",
                    Path(tmp),
                    pii_detector=unreachable,
                )

        def detector(text: str) -> list[PiiEntity]:
            return [PiiEntity("PATIENT", "Anna Mueller")]

        def empty_structure(messages, system_prompt):
            return {"timeline": [], "problems": [], "meds": [], "labs": []}

        with TemporaryDirectory() as tmp:
            document_path = ingest_patient_document(
                config,
                "case-1",
                "Patient: Anna Mueller",
                Path(tmp),
                pii_detector=detector,
                structure_model=empty_structure,
            )
            patient_text = document_path.read_text(encoding="utf-8")
            self.assertIn("[PATIENT_1]", patient_text)
            self.assertNotIn("Anna Mueller", patient_text)

    def test_ollama_detector_parses_local_model_entities(self) -> None:
        calls = []
        shapes = [
            json.dumps({"entities": [{"kind": "PERSON", "value": "Frau Mueller"}]}),
            "```json\n{\"entities\":[{\"kind\":\"ADDRESS\",\"value\":\"Hauptstrasse 7\"}]}\n```",
            json.dumps([{"kind": "EMAIL", "value": "frau@example.test"}]),
            json.dumps({"kind": "PHONE", "value": "+491234"}),
        ]

        def fake_fetch(url: str, payload: dict, timeout_s: float) -> str:
            calls.append((url, payload, timeout_s))
            return json.dumps({"response": shapes.pop(0)})

        detector = OllamaPiiDetector(
            model="local-med-ner",
            base_url="http://127.0.0.1:11434",
            timeout_s=321,
            fetch=fake_fetch,
        )

        entities = []
        for _ in range(4):
            entities.extend(detector("Frau Mueller lebt in Hauptstrasse 7."))

        self.assertEqual(
            entities,
            [
                PiiEntity("PERSON", "Frau Mueller"),
                PiiEntity("ADDRESS", "Hauptstrasse 7"),
                PiiEntity("EMAIL", "frau@example.test"),
                PiiEntity("PHONE", "+491234"),
            ],
        )
        self.assertEqual(calls[0][0], "http://127.0.0.1:11434/api/generate")
        self.assertEqual(calls[0][1]["model"], "local-med-ner")
        self.assertEqual(calls[0][1]["format"]["required"], ["entities"])
        self.assertEqual(calls[0][1]["format"]["properties"]["entities"]["items"]["required"], ["kind", "value"])
        self.assertIn("PERSON includes every named human", calls[0][1]["prompt"])
        self.assertIn("physicians", calls[0][1]["prompt"])
        self.assertIn("Dr. med.", calls[0][1]["prompt"])
        self.assertIn("recipient block", calls[0][1]["prompt"])
        self.assertIn("personal ADDRESS data", calls[0][1]["prompt"])
        self.assertIn("never function words, negations", calls[0][1]["prompt"])
        self.assertIn("Amtsgericht Hamburg", calls[0][1]["prompt"])
        self.assertIs(calls[0][1]["think"], False)
        self.assertEqual(calls[0][1]["options"], {"temperature": 0})
        self.assertEqual({call[2] for call in calls}, {321})

        configured = _ollama_detector(
            Config.model_validate(
                {"deidentification": {"ollama_model": "gemma3:4b", "timeout_s": 123, "chunk_size": 777, "chunk_overlap": 111}}
            )
        )
        self.assertEqual(configured.timeout_s, 123)
        self.assertEqual(configured.chunk_size, 777)
        self.assertEqual(configured.chunk_overlap, 111)

    def test_ollama_detector_unions_overlapping_chunks(self) -> None:
        calls = []

        def fake_fetch(url: str, payload: dict, timeout_s: float) -> str:
            chunk = payload["prompt"].split("TEXT:\n", 1)[1]
            calls.append(chunk)
            entities = []
            if "Alice" in chunk:
                entities.append({"kind": "PERSON", "value": "Alice"})
            if "Bob" in chunk:
                entities.append({"kind": "PERSON", "value": "Bob"})
            return json.dumps({"response": json.dumps({"entities": entities})})

        detector = OllamaPiiDetector(model="local-med-ner", chunk_size=12, chunk_overlap=4, fetch=fake_fetch)
        entities = detector("Alice 123456 Bob 123456 Alice")

        self.assertGreater(len(calls), 1)
        self.assertEqual(entities, [PiiEntity("PERSON", "Alice"), PiiEntity("PERSON", "Bob")])

    def test_ollama_detector_drops_generic_and_one_letter_person_entities(self) -> None:
        calls = []

        def fake_fetch(url: str, payload: dict, timeout_s: float) -> str:
            calls.append(payload)
            return json.dumps({"response": json.dumps({"entities": [
                {"kind": "PERSON", "value": "Eltern"},
                {"kind": "PERSON", "value": "B"},
                {"kind": "PERSON", "value": "Dr. Ingrid Vasquez-Moreno"},
            ]})})

        entities = OllamaPiiDetector(model="local-med-ner", fetch=fake_fetch)("Eltern und Dr. Ingrid")

        self.assertEqual(entities, [PiiEntity("PERSON", "Dr. Ingrid Vasquez-Moreno")])
        self.assertIn("Extract proper names only", calls[0]["prompt"])
        self.assertIn("generic role words like Eltern, Mutter, Patient", calls[0]["prompt"])

    def test_stage1_smoke_reports_pii_boundary(self) -> None:
        with TemporaryDirectory() as tmp:
            report = stage1_smoke(Path(tmp))

            self.assertTrue(report["passed"])
            self.assertEqual(report["patient_id"], "synthetic-stage1")
            self.assertTrue(report["vault_tokens"])
            self.assertTrue(report["case_review"]["evidence_refs"])


if __name__ == "__main__":
    unittest.main()
