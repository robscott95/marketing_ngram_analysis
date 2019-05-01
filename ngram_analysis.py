'''
Script for performing a range n-gram split on passed data.

TODO:
* Clear input text out of stop signs, with respect of dynamic keyword insertation (DKI) and emoji support.
** DKI in Google will require an additionall step, where anything between curly braces {} should have their spaces removed.
* Range n-gram splitting
* Create a post-processing DataFrame containing the original dataframe with new columns containing ngrams
* Create n-gram based DataFrames which sums everything up with respect towards the string based columns.
* Combine all of the n-gram based DF's and post-processing DataFrame into a single dictionary that will be 
  saved into an .xlsx file.
* Support multi-processing
* Support folder parsing
* Support automatic downloading of data from Facebook and Google Ads.
'''

import pandas as pd
import nltk
import argparse


def clean_input_data(input_data_df):
    '''
    Helper function for cleaning the main text column (by default it's the first one).
    Deletes stop characters and in the event of Dynamic Keyword Insertations removes spaces
    between the words inside of curly braces.

    Args:
        - input_data_df (DataFrame): The DF in which the first column contains the text that
            will be tokenized.
    
    Returns:
        - DataFrame: Modified `input_data_df` which has been approprietly pre-processed for 
            further data analysis.

    Raises:
        - TypeError: When the first column of the DataFrame isn't an object.
    '''
    
    if not input_data_df.iloc[0].dtype == 'O':
        raise TypeError(f"The first column of the input file is not text based.")

    stop_characters = ".,:;?!()"

    
    return input_data_df

###################
# MAIN EXECUTABLE #
###################

def execute_ngram_analysis(input_file):
    
    try:
        input_data_df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Reading {input_file} has caused an error:\n{e}")
        return None
    
    input_data_df = clean_input_data(input_data_df)
    

    pass

#####################
# CLI FUNCTIONALITY #
#####################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
    Scripts that analyzes text information via n-gram analysis, and creates
    Excel workbook(s) with said analysis.
    """)

    parser.add_argument('input_file', type=str, help="""
    Relative path to input file that should be analyzed.
    """)

    args = parser.parse_args()

    execute_ngram_analysis(args.input_file)

