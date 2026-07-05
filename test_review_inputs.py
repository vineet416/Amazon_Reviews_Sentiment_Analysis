import unittest

import pandas as pd

from review_inputs import find_text_column, normalize_review_dataframe


class ReviewInputTests(unittest.TestCase):
    def test_xquik_tweet_text_column_becomes_sentence(self):
        df = pd.DataFrame(
            {
                "Tweet Text": [
                    "Great audio quality",
                    "",
                    "Great audio quality",
                    "Setup failed twice",
                ],
                "Tweet Created At": ["2026-07-05"] * 4,
            }
        )

        normalized = normalize_review_dataframe(df)

        self.assertEqual(
            normalized["Sentence"].tolist(),
            ["Great audio quality", "Setup failed twice"],
        )

    def test_alias_lookup_is_case_insensitive(self):
        self.assertEqual(find_text_column(["created_at", "feedback"]), "feedback")

    def test_missing_text_column_raises_clear_error(self):
        df = pd.DataFrame({"score": [1]})

        with self.assertRaisesRegex(ValueError, "text column"):
            normalize_review_dataframe(df)


if __name__ == "__main__":
    unittest.main()
