"""
"""

import unittest
import ngram_analysis

import pandas as pd
import pandas.util.testing


class CleanInputDataTest(unittest.TestCase):
    def test_digits_after_cleaning_being_togheter(self):
        test_df = pd.DataFrame({"description": ["Number is 1 800 800", "$200,000.45"]})
        result_df = ngram_analysis.clean_input_data(test_df)
        test_series = result_df["cleaned_text"]

        assert_series = pd.Series(
            ["number is 1800800", "$20000045"], name="cleaned_text"
        )

        pandas.util.testing.assert_series_equal(test_series, assert_series)

    def test_first_column_is_non_numeric_then_raise_error(self):
        test_df = pd.DataFrame(
            {"non_text_column": [8000, 200], "description": ["spam", "ham"]}
        )

        with self.assertRaises(TypeError):
            ngram_analysis.clean_input_data(test_df)

    def test_for_no_multiple_spaces_present(self):
        test_df = pd.DataFrame(
            {"description": ["Num.ber ... is 1 800 800", "$200,000...45"]}
        )
        result_df = ngram_analysis.clean_input_data(test_df)
        test_series = result_df["cleaned_text"]

        assert_series = pd.Series(
            ["num ber is 1800800", "$20000045"], name="cleaned_text"
        )

        pandas.util.testing.assert_series_equal(test_series, assert_series)

    def test_for_special_keywords(self):
        test_df = pd.DataFrame(
            {
                "description": [
                    "Stress Free Planning For Your Big Day! Customized & All Inclusive Packages. |  no_description2",
                    "{=VenuePrice.Venue} Venue Packages Starting At {=VenuePrice.Start_price} Per Person.\nLearn More!",
                    "{KeyWord:Exceptional Services}. $90,000 |  Views of Vegas Strip Skyline & Sunrise Mountain Range!",
                    "Tie The Knot At One Of The Best Weddings'",
                ]
            }
        )
        result_df = ngram_analysis.clean_input_data(test_df)
        test_series = result_df["cleaned_text"]

        assert_series = pd.Series(
            [
                "stress free planning for your big day customized & all inclusive packages | no_description2",
                "{=venuepricevenue} venue packages starting at {=venuepricestart_price} per person learn more ",
                "{keywordexceptionalservices} $90000 | views of vegas strip skyline & sunrise mountain range ",
                "tie the knot at one of the best weddings'",
            ],
            name="cleaned_text",
        )

        pandas.util.testing.assert_series_equal(test_series, assert_series)

    def test_for_emoji_support(self):
        test_df = pd.DataFrame(
            {
                "description": [
                    "Top marks for customer communication and our test_company app 🤓",
                    "Get great exchange rates abroad, and no fees on card payments! 🙌",
                    "Freeze it in seconds to keep it safe ❄️\nDefrost it if you find it again 🔥",
                ]
            }
        )
        result_df = ngram_analysis.clean_input_data(test_df)
        test_series = result_df["cleaned_text"]

        assert_series = pd.Series(
            [
                "top marks for customer communication and our test_company app 🤓",
                "get great exchange rates abroad and no fees on card payments 🙌",
                "freeze it in seconds to keep it safe ❄️ defrost it if you find it again 🔥",
            ],
            name="cleaned_text",
        )

        pandas.util.testing.assert_series_equal(test_series, assert_series)


class CreateNgramsTest(unittest.TestCase):
    def test_return_target_pattern(self):
        test_df = pd.DataFrame(
            {
                "cleaned_text": [
                    "{keywordexceptionalservices} $90000 | views of vegas skyline & sunrise ",
                    "modern money management",
                    "lost your card freeze it in seconds to keep it safe ❄️",
                ]
            }
        )
        assert_df = test_df
        assert_df["1-gram"] = [
            {
                "{keywordexceptionalservices}",
                "$90000",
                "|",
                "views",
                "of",
                "vegas",
                "skyline",
                "&",
                "sunrise",
            },
            {"management", "money", "modern"},
            {
                "it",
                "card",
                "seconds",
                "❄️",
                "freeze",
                "lost",
                "your",
                "to",
                "keep",
                "safe",
                "in",
            },
        ]
        assert_df["2-gram"] = [
            {
                "{keywordexceptionalservices} $90000",
                "$90000 |",
                "| views",
                "views of",
                "of vegas",
                "vegas skyline",
                "skyline &",
                "& sunrise",
            },
            {"modern money", "money management"},
            {
                "in seconds",
                "safe ❄️",
                "to keep",
                "it in",
                "card freeze",
                "lost your",
                "it safe",
                "freeze it",
                "your card",
                "keep it",
                "seconds to",
            },
        ]
        assert_df["3-gram"] = [
            {
                "{keywordexceptionalservices} $90000 |",
                "$90000 | views",
                "| views of",
                "views of vegas",
                "of vegas skyline",
                "vegas skyline &",
                "skyline & sunrise",
            },
            {"modern money management"},
            {
                "freeze it in",
                "your card freeze",
                "in seconds to",
                "card freeze it",
                "seconds to keep",
                "lost your card",
                "keep it safe",
                "it safe ❄️",
                "to keep it",
                "it in seconds",
            },
        ]
        assert_df["4-gram"] = [
            {
                "{keywordexceptionalservices} $90000 | views",
                "$90000 | views of",
                "| views of vegas",
                "views of vegas skyline",
                "of vegas skyline &",
                "vegas skyline & sunrise",
            },
            set(),
            {
                "lost your card freeze",
                "in seconds to keep",
                "seconds to keep it",
                "freeze it in seconds",
                "to keep it safe",
                "your card freeze it",
                "card freeze it in",
                "it in seconds to",
                "keep it safe ❄️",
            },
        ]

        return_df = ngram_analysis.create_ngrams(test_df, start=1, end=4)

        pandas.util.testing.assert_frame_equal(test_df, return_df)


if __name__ == "__main__":
    unittest.main()
