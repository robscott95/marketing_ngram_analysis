"""
Script for performing a range n-gram split on passed data.

TODO:
* Support multi-processing
* Support automatic downloading of data from Facebook and Google Ads.

* Mention in readme to run `python -m spacy download en` as requirement
* Fix documentation for lemmatization feature
* Known issues - https://github.com/explosion/spaCy/issues/3665
"""

import pandas as pd
import nltk
import spacy
import argparse
import re
import os

from spacy.tokenizer import Tokenizer

pd.options.mode.chained_assignment = None

##############################
# CLEANING AND PREPROCESSING #
##############################


def clean_input_data(input_data_df, lemmatize=False):
    """Helper function for cleaning the main text column
    (default: first one).

    Deletes stop characters. And in the event of numbers or
    Dynamic Keyword Insertations - removes spaces between the words
    inside of curly braces or spaces between digits.

    Supports lemmatization.

    Args:
        - input_data_df (DataFrame): The DF in which the first column
            contains the text that will be tokenized.
        - lemmatize (bool, optional): If set to True the cleaned data
            will also be very conservatively lemmatized, trying to
            normalize only words that are more or less sure with
            omitting word that have special characters in them.
            Defaults to False.

    Returns:
        - DataFrame: Modified `input_data_df` which has now an added
            column `cleaned_text` which contains the processed and
            cleaned text.

    Raises:
        - TypeError: When the first column of the DataFrame isn't an
            object.

    Notes:
        - Current version of Spacy (v2.1.4) is known to have some issues
            with lemmatization
            see (https://github.com/explosion/spaCy/issues/3665).
            So please keep in mind it's more of "experimental" although
            still quite useful.
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

    # TODO: Rewrite this section using spacy's functions
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
    # We just want to remove the most popular punctuation to remove
    # redundant duplicate ngrams while also want to have an insight
    # into how less common characters influence the performance.
    stop_characters = '.,:;?!()"'

    input_data_df["cleaned_text"] = input_data_df["cleaned_text"].str.lower()

    input_data_df["cleaned_text"].replace(
        {f"[{stop_characters}]": " "}, inplace=True, regex=True
    )
    input_data_df["cleaned_text"] = input_data_df["cleaned_text"].apply(
        lambda s: delete_spaces_in_substrings(s)
    )

    input_data_df["cleaned_text"].replace({r"\s+": " "}, inplace=True, regex=True)

    if lemmatize:
        print("Lemmatizing the cleaned text...")
        nlp = spacy.load("en")
        # We don't want to seperate anything that wasn't specified
        # in stop_characters
        supress_re = re.compile(r"""[\.]""")
        nlp.tokenizer = Tokenizer(
            nlp.vocab,
            infix_finditer=supress_re.finditer,
            suffix_search=supress_re.search,
            prefix_search=supress_re.search,
        )

        # Weird bug present in current v2.1.4 of Spacy fix
        nlp.tokenizer.add_special_case(
            "who's", [{spacy.attrs.ORTH: "who's", spacy.attrs.LEMMA: "who's"}]
        )

        input_data_df["cleaned_text"] = input_data_df["cleaned_text"].apply(
            lambda s: " ".join([word.lemma_ for word in nlp(s)])
        )

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

        # set(nltk.ngrams(...)) returns a tuple of ngrams, that's why
        # we join them.
        input_data_cleaned_df[n_gram] = input_data_cleaned_df["cleaned_text"].apply(
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

        print(f"Calculation of {ngram} done.")

    ngram_performance_dict["Original Processed Data"] = input_data_with_ngrams_df.drop(
        "index", axis=1
    )

    return ngram_performance_dict


###################
# MAIN EXECUTABLE #
###################


def execute_ngram_analysis(
    input_file,
    output_folder="ngram_analysis",
    output_file_prefix="Analysis of ",
    lemmatize=False,
):
    """The main function that takes in the path to the .csv with raw
    data and returns the dict containing performance for each ngram.

    Saves the analysis to .xlsx file.

    Args:
        - input_file (str): The relative path to the raw data file in a
            csv format.
        - output_folder (str, optional): The relative path to which the
            output .xlsx files should be written.
            Defaults to "ngram_analysis"
        - output_file_prefix (str, optional): The prefix that will be
            attached to the .xlsx file containing the analysis.
        - lemmatize (bool, optional): If set to True the cleaned data
            will also be very conservatively lemmatized, trying to
            normalize only words that are more or less sure with
            omitting word that have special characters in them.
            Defaults to False.

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
        if lemmatize:
            spacy.load("en")
    except OSError as e:
        print(e)
        print(
            "Please make sure that you've ran `python -m spacy download en` via the console"
        )
        return None

    print(f"\nReading {input_file}")
    try:
        input_data_df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Reading {input_file} has caused an error:\n{e}")
        return None

    print("Cleaning and processing input data...")
    input_data_cleaned_df = clean_input_data(input_data_df, lemmatize=lemmatize)
    input_data_with_ngrams_df = create_ngrams(input_data_cleaned_df)
    print("File cleaning and processing done...")

    print("Calculating performance...")
    ngram_performance_dict = calculate_ngram_performance(input_data_with_ngrams_df)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    input_filename = os.path.splitext(os.path.basename(input_file))[0]
    output_file = f"{output_file_prefix}{input_filename}.xlsx"
    full_output_path = os.path.join(output_folder, output_file)
    print(f"Calculating performance's done. Saving to {full_output_path}")
    with pd.ExcelWriter(full_output_path) as writer:
        for ngram, performance_df in ngram_performance_dict.items():
            performance_df.to_excel(writer, sheet_name=ngram, index=False)
    print(f"Saved to file successfully.")

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

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-file",
        "--input-file",
        type=str,
        help="""
        Relative path to the CSV file containing performance
        data.
        """,
    )
    group.add_argument(
        "-folder",
        "--input-folder",
        type=str,
        help="""
        Relative path to the input folder containing
        only CSV's of raw performance data.
        """,
    )

    parser.add_argument(
        "--lemmatize",
        action="store_true",
        help="""
        Lemmatize grams, converting each one of them to its base form.

        For example rocks will become rock, and computed will become compute.
        """,
    )

    args = parser.parse_args()

    if args.input_folder:
        for filename in os.listdir(args.input_folder):
            file_location = os.path.join(args.input_folder, filename)
            execute_ngram_analysis(file_location, lemmatize=args.lemmatize)
    elif args.input_file:
        execute_ngram_analysis(args.input_file, lemmatize=args.lemmatize)

    print("\nAll done!")
