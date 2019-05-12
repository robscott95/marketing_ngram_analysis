"""
Script for performing a range n-gram split on passed data.

TODO:
* Support folder parsing
* Add an optional argument for stemming (LancasterStemmer might be useful)
* Support multi-processing
* Support automatic downloading of data from Facebook and Google Ads.
"""

import pandas as pd
import nltk
import argparse
import re

pd.options.mode.chained_assignment = None

##############################
# CLEANING AND PREPROCESSING #
##############################


def clean_input_data(input_data_df):
    """Helper function for cleaning the main text column
    (default: first one).

    Deletes stop characters. And in the event of numbers or
    Dynamic Keyword Insertations - removes spaces between the words
    inside of curly braces or spaces between digits.

    Args:
        - input_data_df (DataFrame): The DF in which the first column
            contains the text that will be tokenized.

    Returns:
        - DataFrame: Modified `input_data_df` which has now an added
            column `cleaned_text` which contains the processed and
            cleaned text.

    Raises:
        - TypeError: When the first column of the DataFrame isn't an
            object.
    """

    def delete_spaces_in_substrings(s, pat=r"{.*?}|\d[\d ]*\d"):
        """Helper inner function for removing spaces in a substring of
        a given string. Substrings are determined by the pat argument,
        which is a regex pattern.

        Examples:
            >>> from ngram_analysis import delete_spaces_in_substrings
            >>> test = "test stuff {=venueprice venue} and 1 800 800"
            >>> delete_spaces_in_substrings(test)
            "test stuff {=venuepricevenue} and 1800800"
        """
        matches = re.findall(pat, s)

        if matches:
            for match in matches:
                match_replacement = re.sub(r"\s+", "", match)
                s = re.sub(match, match_replacement, s)

        return s

    # -----------------------
    # Parent's Function Logic
    # -----------------------

    if not input_data_df.iloc[:, 0].dtype == "O":
        raise TypeError(f"The first column of the input file is not text based.")

    # Used in counting how many times a given keyword occured
    input_data_df["Unique Occurences"] = 1
    cols = input_data_df.columns.tolist()
    cols.insert(1, cols.pop(cols.index("Unique Occurences")))
    input_data_df = input_data_df[cols]

    input_data_df["cleaned_text"] = input_data_df.iloc[:, 0]

    input_data_df = input_data_df[pd.notnull(input_data_df.iloc[:, 0])]

    # Leave out curly braces and | sign which denotes DKI and the end of
    # the headline/description in AdWords.
    # We just want to remove the most popular punctuation to remove redundant
    # duplicate ngrams while also want to have an insight into how less
    # common characters influence the performance.
    stop_characters = '.,:;?!()"'

    input_data_df["cleaned_text"] = input_data_df["cleaned_text"].str.lower()

    input_data_df["cleaned_text"].replace(
        {f"[{stop_characters}]": " "}, inplace=True, regex=True
    )
    input_data_df["cleaned_text"] = input_data_df["cleaned_text"].apply(
        lambda s: delete_spaces_in_substrings(s)
    )

    input_data_df["cleaned_text"].replace({r"\s+": " "}, inplace=True, regex=True)

    return input_data_df


def create_ngrams(input_data_cleaned_df, start=1, end=4):
    """Helper function for creating n-grams.
    n is range between start and end (inclusive).

    Examples:
        `df` contains `cleaned_text` and inside - "jack and jill"

        >>> from ngram_analysis import create_ngrams
        >>> create_ngrams(df.iloc[0])
        df["1-gram"]: {"jack", "and", "jill"}
        df["2-gram"]: {"jack and", "and jill"}
        df["3-gram"]: {"jack and jill"}
        df["4-gram"]: set()
    """

    for n in range(start, end + 1):
        n_gram = f"{n}-gram"
        input_data_cleaned_df[n_gram] = input_data_cleaned_df["cleaned_text"].apply(
            # set(nltk.ngrams(...)) returns a tuple of ngrams, that's why we join them.
            lambda s: set(" ".join(gram) for gram in set(nltk.ngrams(s.split(), n)))
        )

    return input_data_cleaned_df


###########################
# CALCULATING PERFORMANCE #
###########################


def calculate_ngram_performance(input_data_with_ngrams_df):
    """Helper function which aggregates the unique n-grams and apply's
    two different types of aggregation depending if the performance
    data is text or numerical based.

    Args:
        - input_data_with_ngrams_df (DataFrame): DataFrame containing
            the text, performance columns, cleaned text and ngram
            columns.

    Returns:
        - dict: A dictionary containing key value pairs of the ngram
            and ngram's performance DataFrame.

            {
                "1-gram": DataFrame({
                    "1-gram": ["jack", "and", "jill"],
                    "link_clicks": [1000, 3000, 2000],
                    "in_ads": ["ad_1", "ad_1, ad_2", "ad_2"]
                }),
                "2-gram": DataFrame({
                    "2-gram": ["jack and", "and jill"],
                    "link_clicks": [1000, 2000],
                    "in_ads": ["ad_1", "ad_2"]
                })
            }
    """

    def aggregate_by_dtype(x):
        """Inner helper function for the groupby aggregation

        Note:
            - The aggregated returned text has only unique elements
                seperated by a ",".
        """
        d = {}
        for column in x.columns[1:]:
            if x[column].dtype == "O":
                d[column] = ", ".join(set(x[column]))
            else:
                d[column] = x[column].sum()
        return pd.Series(d)

    # -----------------------
    # Parent's Function Logic
    # -----------------------

    input_df_columns = input_data_with_ngrams_df.columns.tolist()
    ngram_columns = [col for col in input_df_columns if "-gram" in col]
    performance_columns = [
        col
        for col in input_df_columns[1:]
        if (col not in ngram_columns) and (col != "cleaned_text")
    ]

    input_data_with_ngrams_df.reset_index(level=0, inplace=True)  # Key for merging
    merging_columns = performance_columns.copy()
    merging_columns.append("index")

    ngram_performance_dict = {}

    for ngram in ngram_columns:
        # This returns us a DataFrame which has n-gram keyword
        # in one column, and the original index (key) in the other.
        id_and_ngram_df = (
            pd.DataFrame(input_data_with_ngrams_df[ngram].values.tolist())
            .reset_index(level=0)
            .melt(id_vars=["index"], value_name=ngram)
            .dropna()
            .drop(columns=["variable"])
        )

        ngram_performance_df = id_and_ngram_df.merge(
            input_data_with_ngrams_df[merging_columns], on="index"
        ).drop(columns=["index"])

        ngram_performance_df = (
            ngram_performance_df.groupby(ngram).apply(aggregate_by_dtype).reset_index()
        )
        ngram_performance_dict[ngram] = ngram_performance_df

    ngram_performance_dict["Original Processed Data"] = input_data_with_ngrams_df.drop(
        "index", axis=1
    )

    return ngram_performance_dict


###################
# MAIN EXECUTABLE #
###################


def execute_ngram_analysis(input_file):
    """The main function that takes in the path to the .csv with raw
    data and returns the dict containing performance for each ngram.
    Also saves to a file.

    Args:
        - input_file (str): The relative path to the raw data file in a
            csv format.

    Returns:
        - dict: A dictionary containing key value pairs of the ngram
            and ngram's performance DataFrame.

            {
                "1-gram": DataFrame({
                    "1-gram": ["jack", "and", "jill"],
                    "link_clicks": [1000, 3000, 2000],
                    "in_ads": ["ad_1", "ad_1, ad_2", "ad_2"]
                }),
                "2-gram": DataFrame({
                    "2-gram": ["jack and", "and jill"],
                    "link_clicks": [1000, 2000],
                    "in_ads": ["ad_1", "ad_2"]
                })
            }

    Notes:
        - Also saves the output to an .xlsx formatted file.
        - Requirements for the input .csv file:
            * The first column is required to be the text that you
                wish to be analyzed, rest of the columns is
                performance data.
            * The performance data shouldn't be calculated in any
                way (averages, ROI, etc.) as thiswill simply
                return nonsense data due to summing them up.
    """

    try:
        input_data_df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Reading {input_file} has caused an error:\n{e}")
        return None

    print("Cleaning and processing input data...")
    input_data_cleaned_df = clean_input_data(input_data_df)
    input_data_with_ngrams_df = create_ngrams(input_data_cleaned_df)
    print("File cleaning and processing done...")

    print("Calculating performance...")
    ngram_performance_dict = calculate_ngram_performance(input_data_with_ngrams_df)

    output_file = "output.xlsx"
    print(f"Calculating performance's done. Saving to {output_file}")
    with pd.ExcelWriter(output_file) as writer:
        for ngram, performance_df in ngram_performance_dict.items():
            performance_df.to_excel(writer, sheet_name=ngram, index=False)

    return ngram_performance_dict


#####################
# CLI FUNCTIONALITY #
#####################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
    Scripts that analyzes text information via n-gram analysis, and creates
    Excel workbook(s) with said analysis.
    """
    )

    parser.add_argument(
        "input_file",
        type=str,
        help="""
    Relative path to input file that should be analyzed.
    """,
    )

    args = parser.parse_args()

    execute_ngram_analysis(args.input_file)
