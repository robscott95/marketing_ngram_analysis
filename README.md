# n-gram Analysis for Marketing
A python based ngram analysis script which allows easy analysis of text data (adwords search terms, ad copy, etc).

## Project plan
* The script in the initial version will only use a pre-cleaned .csv file, in which the only requirement is that
the first column should be the text that should be tokenized (the headline, description, or whatever). The rest
of the columns will contain any metric that one ones to count or text information somebody wants to add up 
(like the Headline ID or Ad ID to know in which ads given n-gram was present)
  * This will allow for some flexibility in choosing the columns, but a best-practice kind of file or README
 section should be created.
  * Tokenization should allow for Dynamic Keyword Insertation and Emoji support.
* The output file should be an .xlsx workbook, containing sheets with selected ngram range (1-gram, 2-gram, ..., n-gram)
and a sheet with pre-processed data, so one can debug weird occurences. Especially helpful with Ad ID.
* Utilize multithreading to speed things up.
* In later stages, might be worth expanding it to automatically download data from Google Ads and Facebook Ads.
