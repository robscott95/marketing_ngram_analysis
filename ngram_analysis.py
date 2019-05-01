'''
Script for performing a range n-gram split on passed data.

TODO:
* Clear input text out of stop signs, with respect of dynamic keyword insertation (DKI) and emoji support.
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

