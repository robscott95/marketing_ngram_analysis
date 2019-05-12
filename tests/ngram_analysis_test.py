import pytest
import ngram_analysis

import pandas as pd
import pandas.util.testing


class TestCleanInputData:
    def test_digits_after_cleaning_being_togheter(self):
        test_df = pd.DataFrame({"description": ["Number is 1 800 800", "$200,000.45"]})
        result_df = ngram_analysis.clean_input_data(test_df)
        result_series = result_df["cleaned_text"]

        assert_series = pd.Series(
            ["number is 1800800", "$20000045"], name="cleaned_text"
        )

        pandas.util.testing.assert_series_equal(result_series, assert_series)

    def test_if_first_column_is_non_numeric_then_raise_error(self):
        test_df = pd.DataFrame(
            {"non_text_column": [8000, 200], "description": ["spam", "ham"]}
        )

        with pytest.raises(TypeError):
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
        result_series = result_df["cleaned_text"]

        assert_series = pd.Series(
            [
                "stress free planning for your big day customized & all inclusive packages | no_description2",
                "{=venuepricevenue} venue packages starting at {=venuepricestart_price} per person learn more ",
                "{keywordexceptionalservices} $90000 | views of vegas strip skyline & sunrise mountain range ",
                "tie the knot at one of the best weddings'",
            ],
            name="cleaned_text",
        )

        pandas.util.testing.assert_series_equal(result_series, assert_series)

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
        result_series = result_df["cleaned_text"]

        assert_series = pd.Series(
            [
                "top marks for customer communication and our test_company app 🤓",
                "get great exchange rates abroad and no fees on card payments 🙌",
                "freeze it in seconds to keep it safe ❄️ defrost it if you find it again 🔥",
            ],
            name="cleaned_text",
        )

        pandas.util.testing.assert_series_equal(result_series, assert_series)


class TestCreateNgrams:
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

        pandas.util.testing.assert_frame_equal(return_df, assert_df)

    def test_return_only_unique_grams(self):
        test_df = pd.DataFrame(
            {
                "cleaned_text": [
                    "modern money money management",
                    "lost your card card is lost",
                ]
            }
        )

        assert_df = test_df
        assert_df["1-gram"] = [
            {"modern", "money", "management"},
            {"lost", "your", "card", "is"},
        ]
        assert_df["2-gram"] = [
            {"modern money", "money management", "money money"},
            {"lost your", "your card", "card card", "card is", "is lost"},
        ]
        assert_df["3-gram"] = [
            {"modern money money", "money money management"},
            {"lost your card", "your card card", "card card is", "card is lost"},
        ]
        assert_df["4-gram"] = [
            {"modern money money management"},
            {"lost your card card", "your card card is", "card card is lost"},
        ]

        return_df = ngram_analysis.create_ngrams(test_df, start=1, end=4)

        pandas.util.testing.assert_frame_equal(return_df, assert_df)


class TestCalculateNgramsPerformance:
    def test_numerical_aggregation(self):
        test_input_df = pd.DataFrame(
            {
                "cleaned_text": [
                    "jack and jill made money",
                    "jill and bart made money",
                ],
                "link_clicks": [1000, 2000],
                "1-gram": [
                    {"and", "jack", "money", "made", "jill"},
                    {"and", "money", "made", "bart", "jill"},
                ],
                "2-gram": [
                    {"made money", "and jill", "jack and", "jill made"},
                    {"made money", "jill and", "and bart", "bart made"},
                ],
                "3-gram": [
                    {"jack and jill", "jill made money", "and jill made"},
                    {"bart made money", "and bart made", "jill and bart"},
                ],
                "4-gram": [
                    {"and jill made money", "jack and jill made"},
                    {"and bart made money", "jill and bart made"},
                ],
            }
        )

        assert_dict = {
            "1-gram": pd.DataFrame(
                {
                    "1-gram": ["and", "bart", "jack", "jill", "made", "money"],
                    "link_clicks": [3000, 2000, 1000, 3000, 3000, 3000],
                }
            ),
            "2-gram": pd.DataFrame(
                {
                    "2-gram": [
                        "and bart",
                        "and jill",
                        "bart made",
                        "jack and",
                        "jill and",
                        "jill made",
                        "made money",
                    ],
                    "link_clicks": [2000, 1000, 2000, 1000, 2000, 1000, 3000],
                }
            ),
            "3-gram": pd.DataFrame(
                {
                    "3-gram": [
                        "and bart made",
                        "and jill made",
                        "bart made money",
                        "jack and jill",
                        "jill and bart",
                        "jill made money",
                    ],
                    "link_clicks": [2000, 1000, 2000, 1000, 2000, 1000],
                }
            ),
            "4-gram": pd.DataFrame(
                {
                    "4-gram": [
                        "and bart made money",
                        "and jill made money",
                        "jack and jill made",
                        "jill and bart made",
                    ],
                    "link_clicks": [2000, 1000, 1000, 2000],
                }
            ),
            "Original Processed Data": pd.DataFrame(
                {
                    "cleaned_text": [
                        "jack and jill made money",
                        "jill and bart made money",
                    ],
                    "link_clicks": [1000, 2000],
                    "1-gram": [
                        {"and", "jack", "money", "made", "jill"},
                        {"and", "money", "made", "bart", "jill"},
                    ],
                    "2-gram": [
                        {"made money", "and jill", "jack and", "jill made"},
                        {"made money", "jill and", "and bart", "bart made"},
                    ],
                    "3-gram": [
                        {"jack and jill", "jill made money", "and jill made"},
                        {"bart made money", "and bart made", "jill and bart"},
                    ],
                    "4-gram": [
                        {"and jill made money", "jack and jill made"},
                        {"and bart made money", "jill and bart made"},
                    ],
                }
            ),
        }

        return_dict = ngram_analysis.calculate_ngram_performance(test_input_df)

        # Transforming inner DataFrames into dicts because we can't easily
        # compare dicts with DF's in them.
        return_dict = {k: v.to_dict() for k, v in return_dict.items()}
        assert_dict = {k: v.to_dict() for k, v in assert_dict.items()}

        assert return_dict == assert_dict

    def test_text_aggregation(self):
        test_input_df = pd.DataFrame(
            {
                "cleaned_text": [
                    "jack and jill made money",
                    "jill and bart made money",
                ],
                "ad_id": ["ad_1", "ad_2"],
                "1-gram": [
                    {"and", "jack", "money", "made", "jill"},
                    {"and", "money", "made", "bart", "jill"},
                ],
                "2-gram": [
                    {"made money", "and jill", "jack and", "jill made"},
                    {"made money", "jill and", "and bart", "bart made"},
                ],
                "3-gram": [
                    {"jack and jill", "jill made money", "and jill made"},
                    {"bart made money", "and bart made", "jill and bart"},
                ],
                "4-gram": [
                    {"and jill made money", "jack and jill made"},
                    {"and bart made money", "jill and bart made"},
                ],
            }
        )

        assert_dict = {
            "1-gram": pd.DataFrame(
                {
                    "1-gram": ["and", "bart", "jack", "jill", "made", "money"],
                    "ad_id": [
                        "ad_2, ad_1",
                        "ad_2",
                        "ad_1",
                        "ad_2, ad_1",
                        "ad_2, ad_1",
                        "ad_2, ad_1",
                    ],
                }
            ),
            "2-gram": pd.DataFrame(
                {
                    "2-gram": [
                        "and bart",
                        "and jill",
                        "bart made",
                        "jack and",
                        "jill and",
                        "jill made",
                        "made money",
                    ],
                    "ad_id": [
                        "ad_2",
                        "ad_1",
                        "ad_2",
                        "ad_1",
                        "ad_2",
                        "ad_1",
                        "ad_2, ad_1",
                    ],
                }
            ),
            "3-gram": pd.DataFrame(
                {
                    "3-gram": [
                        "and bart made",
                        "and jill made",
                        "bart made money",
                        "jack and jill",
                        "jill and bart",
                        "jill made money",
                    ],
                    "ad_id": ["ad_2", "ad_1", "ad_2", "ad_1", "ad_2", "ad_1"],
                }
            ),
            "4-gram": pd.DataFrame(
                {
                    "4-gram": [
                        "and bart made money",
                        "and jill made money",
                        "jack and jill made",
                        "jill and bart made",
                    ],
                    "ad_id": ["ad_2", "ad_1", "ad_1", "ad_2"],
                }
            ),
            "Original Processed Data": pd.DataFrame(
                {
                    "cleaned_text": [
                        "jack and jill made money",
                        "jill and bart made money",
                    ],
                    "ad_id": ["ad_1", "ad_2"],
                    "1-gram": [
                        {"and", "jack", "money", "made", "jill"},
                        {"and", "money", "made", "bart", "jill"},
                    ],
                    "2-gram": [
                        {"made money", "and jill", "jack and", "jill made"},
                        {"made money", "jill and", "and bart", "bart made"},
                    ],
                    "3-gram": [
                        {"jack and jill", "jill made money", "and jill made"},
                        {"bart made money", "and bart made", "jill and bart"},
                    ],
                    "4-gram": [
                        {"and jill made money", "jack and jill made"},
                        {"and bart made money", "jill and bart made"},
                    ],
                }
            ),
        }

        return_dict = ngram_analysis.calculate_ngram_performance(test_input_df)

        # Transforming inner DataFrames into dicts because we can't easily
        # compare dicts with DF's in them.
        return_dict = {k: v.to_dict() for k, v in return_dict.items()}

        # Creating this second dict assertion as one trivial bug may appear
        # with "ad_2, ad_1" swapping to "ad_1, ad_2".
        assert_dict1 = {k: v.to_dict() for k, v in assert_dict.items()}
        assert_dict2 = {k: v.to_dict() for k, v in assert_dict.items()}  # Deep copy
        assert_dict2["1-gram"]["ad_id"] = {
            0: "ad_1, ad_2",
            1: "ad_2",
            2: "ad_1",
            3: "ad_1, ad_2",
            4: "ad_1, ad_2",
            5: "ad_1, ad_2",
        }
        assert_dict2["2-gram"]["ad_id"] = {
            0: "ad_2",
            1: "ad_1",
            2: "ad_2",
            3: "ad_1",
            4: "ad_2",
            5: "ad_1",
            6: "ad_1, ad_2",
        }

        assert (return_dict == assert_dict1) or (return_dict == assert_dict2)
