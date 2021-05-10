import env
import pandas as pd
import utilities as utils
import re

def prepare_bootcamp_readme_df(df):
    bootcamp_readme_df = df.copy()
    
    bootcamp_readme_df = _drop_nan_empty_rows(bootcamp_readme_df)
    bootcamp_readme_df = _remove_non_target_languages(bootcamp_readme_df)
    bootcamp_readme_df = _make_contents_ready_for_nlp_exloration(bootcamp_readme_df)
    
    return bootcamp_readme_df

def _drop_nan_empty_rows(df):
    bootcamp_readme_df = df.copy()
    
    # The positions dictionary contains the rows and columns of nan values and empty values in the dataframe
    positions = utils.nan_null_empty_check(bootcamp_readme_df)
    
    # A list is compiled from both nan_positions and empty_positions, then using set(), a list with no duplicates is generated
    drop_rows = list(positions['empty_positions'][0]) + list(positions['nan_positions'][0])
    drop_rows = list(set(drop_rows))
    
    return bootcamp_readme_df.drop(index=drop_rows)

def _make_contents_ready_for_nlp_exloration(df):
    bootcamp_readme_df = df.copy()
    
    bootcamp_readme_df.readme_contents = bootcamp_readme_df.readme_contents.apply(utils.nlp_basic_clean)

    # Basic clean did not remove new line characters so remove them here.
    bootcamp_readme_df.readme_contents = bootcamp_readme_df.readme_contents.apply(lambda readme : re.sub(r"\n", " ", readme))

    bootcamp_readme_df.readme_contents = bootcamp_readme_df.readme_contents.apply(utils.nlp_tokenize)
    bootcamp_readme_df.readme_contents = bootcamp_readme_df.readme_contents.apply(utils.nlp_remove_stopwords)
    bootcamp_readme_df.readme_contents = bootcamp_readme_df.readme_contents.apply(utils.nlp_lemmatize)
    
    return bootcamp_readme_df

def _remove_non_target_languages(df):
    bootcamp_readme_df = df.copy()
    
    return bootcamp_readme_df[(bootcamp_readme_df.language == "JavaScript") | (bootcamp_readme_df.language == "Jupyter Notebook") \
                              | (bootcamp_readme_df.language == "Python") | (bootcamp_readme_df.language == "Java")]