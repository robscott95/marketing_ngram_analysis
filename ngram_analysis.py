"""
Script for performing a range n-gram split on passed data.

TODO:
* Range n-gram splitting
* Create a post-processing DataFrame containing the original dataframe with
    new columns containing ngrams
* Create n-gram based DataFrames which sums everything up with respect towards
    the string based columns.
* Combine all of the n-gram based DF's and post-processing DataFrame into a
    single dictionary that will be saved into an .xlsx file.
* Support multi-processing
* Support folder parsing
* Support automatic downloading of data from Facebook and Google Ads.
"""

import pandas as pd
import nltk
import argparse
import re

pd.options.mode.chained_assignment = None


def clean_input_data(input_data_df):
    """Helper function for cleaning the main text column (default: first one).

    Deletes stop characters and in the event of Dynamic Keyword Insertations
    removes spaces between the words inside of curly braces or spaces between
    digits.

    Args:
        - input_data_df (DataFrame): The DF in which the first column
            contains the text that will be tokenized.

    Returns:
        - DataFrame: Modified `input_data_df` which has now an added column
            `cleaned_text` which contains the processed and cleaned text.

    Raises:
        - TypeError: When the first column of the DataFrame isn't an object.
    """

    def delete_spaces_in_substrings(s, pat=r"{.*?}|\d[\d ]*\d"):
        """Helper inner function for removing spaces in a substring of a
        given string. Substrings are determined by the pat argument,
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



###################
# MAIN EXECUTABLE #
###################


def execute_ngram_analysis(input_file):

    try:
        input_data_df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Reading {input_file} has caused an error:\n{e}")
        return None

    input_data_cleaned_df = clean_input_data(input_data_df)
    input_data_with_ngrams_df = create_ngrams(input_data_cleaned_df)
    print("File cleaning and processing done...")


    pass


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
