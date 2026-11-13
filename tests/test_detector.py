import unittest
from pathlib import Path

from detector import TfidfDuplicateDetector, load_tickets, tokenize

DATA = Path(__file__).parents[1] / "data" / "tickets.csv"


class DuplicateTests(unittest.TestCase):
    def test_tokenize(self):
        self.assertEqual(tokenize("CSV export failed!"), ["csv", "export", "failed"])

    def test_loads_tickets(self):
        self.assertEqual(len(load_tickets(DATA)), 8)

    def test_similar_tickets_rank_highly(self):
        detector = TfidfDuplicateDetector().fit(load_tickets(DATA))
        matches = detector.find_duplicates("The invoice CSV export is empty", threshold=0.2)
        self.assertTrue(matches)
        self.assertEqual(matches[0][0], 2)
        self.assertGreater(matches[0][1], 0.4)

    def test_unrelated_query_has_no_match(self):
        detector = TfidfDuplicateDetector().fit(load_tickets(DATA))
        self.assertEqual(detector.find_duplicates("office lunch menu update", threshold=0.35), [])

    def test_invalid_threshold_rejected(self):
        detector = TfidfDuplicateDetector().fit(load_tickets(DATA))
        with self.assertRaises(ValueError):
            detector.find_duplicates("test", threshold=1.5)


if __name__ == "__main__":
    unittest.main()
