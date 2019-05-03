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


if __name__ == "__main__":
    unittest.main()
