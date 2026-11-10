"""Offline tests for invoice tools and posting eligibility checks."""

from __future__ import annotations

import unittest
from decimal import Decimal
from unittest.mock import patch

import invoice_tools
from invoice_tools import (
    BlockedVendor,
    DuplicateInvoice,
    InvalidInvoiceId,
    InvalidInvoiceTotal,
    InvoiceNotFound,
    VendorNotFound,
    list_invoices,
    lookup_vendor,
    prepare_post,
    read_invoice,
    validate_posting_amount,
)


class InvoiceToolsTests(unittest.TestCase):
    def test_list_all_six_invoices(self) -> None:
        rows = list_invoices()
        self.assertEqual(
            [row["invoice_id"] for row in rows],
            [
                "invoice-01",
                "invoice-02",
                "invoice-03",
                "invoice-04",
                "invoice-05",
                "invoice-06",
            ],
        )
        self.assertEqual(rows[0]["invoice_number"], "SGP/26-27/0412")
        self.assertEqual(rows[0]["total"], "16862.20")
        self.assertEqual(rows[5]["total"], "21948.00")

    def test_read_invoice_01(self) -> None:
        invoice = read_invoice("invoice-01")
        self.assertEqual(invoice["invoice_number"], "SGP/26-27/0412")
        self.assertEqual(invoice["vendor_name"], "Shree Ganesh Packaging Pvt Ltd")
        self.assertEqual(invoice["currency"], "INR")
        self.assertEqual(invoice["total"], "16862.20")
        self.assertIsInstance(invoice["total"], str)

    def test_read_invoice_06_preserves_lines_and_total(self) -> None:
        invoice = read_invoice("invoice-06")
        self.assertEqual(len(invoice["lines"]), 1)
        self.assertEqual(invoice["lines"][0]["amount"], "18600.00")
        self.assertEqual(invoice["total"], "21948.00")
        self.assertEqual(invoice["vendor_name"], "Shree Ganesh Packaging Pvt Ltd")

    def test_reject_invalid_ids_and_traversal(self) -> None:
        for bad in (
            "",
            "invoice-01.json",
            "../vendors",
            "../../etc/passwd",
            "invoice-01/../../ledger.py",
            "INVOICE-01",
            "/etc/passwd",
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(InvalidInvoiceId):
                    read_invoice(bad)

    def test_missing_invoice(self) -> None:
        with self.assertRaises(InvoiceNotFound):
            read_invoice("invoice-99")
        # Well-formed pattern but not on the whitelist.
        with self.assertRaises(InvoiceNotFound):
            read_invoice("invoice-1")

    def test_lookup_active_vendor(self) -> None:
        vendor = lookup_vendor(name="Bluefin Logistics LLP")
        self.assertEqual(vendor["vendor_id"], "V-002")
        self.assertEqual(vendor["status"], "active")

        by_id = lookup_vendor(vendor_id="V-001")
        self.assertEqual(by_id["name"], "Shree Ganesh Packaging Pvt Ltd")
        self.assertEqual(by_id["status"], "active")

    def test_lookup_blocked_vendor_includes_reason(self) -> None:
        vendor = lookup_vendor(name="Orbit Facility Services")
        self.assertEqual(vendor["vendor_id"], "V-004")
        self.assertEqual(vendor["status"], "blocked")
        self.assertEqual(vendor["blocked_reason"], "Bank details under verification")

        by_id = lookup_vendor(vendor_id="V-004")
        self.assertEqual(by_id["blocked_reason"], "Bank details under verification")

    def test_unknown_vendor(self) -> None:
        with self.assertRaises(VendorNotFound):
            lookup_vendor(name="Not A Real Vendor")
        with self.assertRaises(VendorNotFound):
            lookup_vendor(vendor_id="V-999")

    @patch("invoice_tools.ledger.entries", return_value=[])
    @patch("invoice_tools.ledger.post")
    def test_prepare_invoice_01_when_ledger_empty(self, mock_post, mock_entries) -> None:
        result = prepare_post("invoice-01")
        self.assertEqual(
            result,
            {
                "invoice_id": "invoice-01",
                "vendor_id": "V-001",
                "vendor_name": "Shree Ganesh Packaging Pvt Ltd",
                "invoice_number": "SGP/26-27/0412",
                "invoice_date": "2026-10-03",
                "currency": "INR",
                "amount": "16862.20",
            },
        )
        mock_entries.assert_called_once_with(
            vendor_id="V-001",
            invoice_number="SGP/26-27/0412",
        )
        mock_post.assert_not_called()

    @patch("invoice_tools.ledger.entries", return_value=[])
    @patch("invoice_tools.ledger.post")
    def test_prepare_invoice_02_when_ledger_empty(self, mock_post, mock_entries) -> None:
        result = prepare_post("invoice-02")
        self.assertEqual(result["vendor_id"], "V-002")
        self.assertEqual(result["invoice_number"], "BL-88213")
        self.assertEqual(result["amount"], "70800.00")
        mock_entries.assert_called_once_with(
            vendor_id="V-002",
            invoice_number="BL-88213",
        )
        mock_post.assert_not_called()

    @patch("invoice_tools.ledger.entries")
    @patch("invoice_tools.ledger.post")
    def test_reject_invoice_04_blocked_vendor(self, mock_post, mock_entries) -> None:
        with self.assertRaises(BlockedVendor) as ctx:
            prepare_post("invoice-04")
        message = str(ctx.exception)
        self.assertIn("V-004", message)
        self.assertIn("Bank details under verification", message)
        mock_entries.assert_not_called()
        mock_post.assert_not_called()

    @patch(
        "invoice_tools.ledger.entries",
        return_value=[
            {
                "entry_id": "JE-000001",
                "vendor_id": "V-002",
                "invoice_number": "BL-88190",
                "amount": "25960.00",
            }
        ],
    )
    @patch("invoice_tools.ledger.post")
    def test_reject_invoice_05_duplicate(self, mock_post, mock_entries) -> None:
        with self.assertRaises(DuplicateInvoice) as ctx:
            prepare_post("invoice-05")
        message = str(ctx.exception)
        self.assertIn("BL-88190", message)
        self.assertIn("JE-000001", message)
        mock_entries.assert_called_once_with(
            vendor_id="V-002",
            invoice_number="BL-88190",
        )
        mock_post.assert_not_called()

    @patch("invoice_tools.ledger.entries")
    def test_duplicate_check_uses_vendor_id_and_invoice_number(self, mock_entries) -> None:
        mock_entries.return_value = []
        prepare_post("invoice-03")
        mock_entries.assert_called_once_with(
            vendor_id="V-003",
            invoice_number="KOS-2026-1187",
        )

    def test_reject_invalid_monetary_totals(self) -> None:
        for bad in ("0", "0.00", "-10.00", "not-a-number", "", None, True):
            with self.subTest(bad=bad):
                with self.assertRaises(InvalidInvoiceTotal):
                    validate_posting_amount(bad)

        with self.assertRaises(InvalidInvoiceTotal):
            validate_posting_amount(Decimal("NaN"))
        with self.assertRaises(InvalidInvoiceTotal):
            validate_posting_amount(Decimal("Infinity"))

        # Isolated prepare_post path: do not alter real fixtures.
        fake = {
            "invoice_number": "FAKE-1",
            "invoice_date": "2026-10-01",
            "vendor_name": "Bluefin Logistics LLP",
            "currency": "INR",
            "total": "-1.00",
        }
        with patch.object(invoice_tools, "read_invoice", return_value=fake):
            with patch("invoice_tools.ledger.entries", return_value=[]):
                with self.assertRaises(InvalidInvoiceTotal):
                    prepare_post("invoice-01")

    @patch("invoice_tools.ledger.post")
    def test_suite_never_calls_ledger_post_on_prepare_paths(self, mock_post) -> None:
        with patch("invoice_tools.ledger.entries", return_value=[]):
            prepare_post("invoice-01")
        with self.assertRaises(BlockedVendor):
            prepare_post("invoice-04")
        with patch(
            "invoice_tools.ledger.entries",
            return_value=[{"entry_id": "JE-000001", "vendor_id": "V-002", "invoice_number": "BL-88190"}],
        ):
            with self.assertRaises(DuplicateInvoice):
                prepare_post("invoice-05")
        mock_post.assert_not_called()

    def test_ledger_lookup_failure_is_not_treated_as_eligible(self) -> None:
        with patch(
            "invoice_tools.ledger.entries",
            side_effect=RuntimeError("ledger unavailable"),
        ):
            with self.assertRaises(RuntimeError):
                prepare_post("invoice-01")

    @patch("invoice_tools.ledger.entries", return_value=[])
    @patch("invoice_tools.ledger.post")
    def test_active_vendor_accepted_for_prepare(self, mock_post, mock_entries) -> None:
        active = {
            "vendor_id": "V-001",
            "name": "Shree Ganesh Packaging Pvt Ltd",
            "status": "active",
        }
        with patch.object(invoice_tools, "lookup_vendor", return_value=active):
            result = prepare_post("invoice-01")
        self.assertEqual(result["vendor_id"], "V-001")
        self.assertEqual(result["amount"], "16862.20")
        mock_entries.assert_called_once()
        mock_post.assert_not_called()

    @patch("invoice_tools.ledger.entries")
    @patch("invoice_tools.ledger.post")
    def test_blocked_vendor_rejected_before_ledger_lookup(self, mock_post, mock_entries) -> None:
        blocked = {
            "vendor_id": "V-004",
            "name": "Orbit Facility Services",
            "status": "blocked",
            "blocked_reason": "Bank details under verification",
        }
        with patch.object(invoice_tools, "lookup_vendor", return_value=blocked):
            with self.assertRaises(BlockedVendor) as ctx:
                prepare_post("invoice-01")
        self.assertIn("Bank details under verification", str(ctx.exception))
        mock_entries.assert_not_called()
        mock_post.assert_not_called()

    @patch("invoice_tools.ledger.entries")
    @patch("invoice_tools.ledger.post")
    def test_unexpected_vendor_status_rejected(self, mock_post, mock_entries) -> None:
        suspended = {
            "vendor_id": "V-001",
            "name": "Shree Ganesh Packaging Pvt Ltd",
            "status": "suspended",
        }
        with patch.object(invoice_tools, "lookup_vendor", return_value=suspended):
            with self.assertRaises(BlockedVendor) as ctx:
                prepare_post("invoice-01")
        self.assertIn("suspended", str(ctx.exception))
        mock_entries.assert_not_called()
        mock_post.assert_not_called()

    @patch("invoice_tools.ledger.entries")
    @patch("invoice_tools.ledger.post")
    def test_missing_vendor_status_rejected(self, mock_post, mock_entries) -> None:
        no_status = {
            "vendor_id": "V-001",
            "name": "Shree Ganesh Packaging Pvt Ltd",
        }
        with patch.object(invoice_tools, "lookup_vendor", return_value=no_status):
            with self.assertRaises(BlockedVendor) as ctx:
                prepare_post("invoice-01")
        self.assertIn("status=None", str(ctx.exception))
        mock_entries.assert_not_called()
        mock_post.assert_not_called()

    def test_existing_fixture_totals_accepted_unchanged(self) -> None:
        self.assertEqual(validate_posting_amount("16862.20"), "16862.20")
        self.assertEqual(validate_posting_amount("70800.00"), "70800.00")
        self.assertEqual(validate_posting_amount("12.3"), "12.3")

    def test_more_than_two_decimal_places_rejected(self) -> None:
        with self.assertRaises(InvalidInvoiceTotal):
            validate_posting_amount("12.345")

    def test_tiny_positive_amount_that_rounds_to_zero_rejected(self) -> None:
        with self.assertRaises(InvalidInvoiceTotal):
            validate_posting_amount("0.001")

    @patch(
        "invoice_tools.ledger.entries",
        return_value=[
            {
                "entry_id": "JE-000001",
                "vendor_id": "V-002",
                "invoice_number": "BL-88190",
            }
        ],
    )
    @patch("invoice_tools.ledger.post")
    def test_duplicate_detection_still_works_after_corrections(self, mock_post, mock_entries) -> None:
        with self.assertRaises(DuplicateInvoice):
            prepare_post("invoice-05")
        mock_entries.assert_called_once_with(
            vendor_id="V-002",
            invoice_number="BL-88190",
        )
        mock_post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
