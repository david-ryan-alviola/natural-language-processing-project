import env
import pandas as pd
import re
import matplotlib.pyplot as plt

from wordcloud import WordCloud

def generate_language_count_df(language_series):
    languages_df = pd.concat([language_series.value_counts(), language_series.value_counts(normalize=True)], axis=1)
    languages_df.columns = ['count', 'percent']
    
    return languages_df

def _generate_dict_of_words_by_language(languages_df, bootcamp_readme_df):
    readme_words_by_language_dict = {}

    for language in languages_df.index:
        all_readme_for_language = " ".join(bootcamp_readme_df[bootcamp_readme_df.language == language].readme_contents)
    
        # The dictionary entry for a language will hold the list of unique words in all the readme files
        readme_words_by_language_dict[language] = re.sub(r"[^\w\s]", "", all_readme_for_language).split()
        
    return readme_words_by_language_dict

def _generate_dict_of_word_counts_by_language(readme_words_by_language_dict):
    word_count_by_lanugage_dict = {}

    for language in readme_words_by_language_dict.keys():
        # The dictionary entry for a language will hold the count of occurences for each unique word in all the readme files for that language
        word_count_by_lanugage_dict[language] = pd.Series(readme_words_by_language_dict[language]).value_counts()
        
    return word_count_by_lanugage_dict

def generate_word_list_and_count_dictionaries(languages_df, bootcamp_readme_df):
    readme_words_by_language_dict = _generate_dict_of_words_by_language(languages_df, bootcamp_readme_df)
    word_count_by_lanugage_dict = _generate_dict_of_word_counts_by_language(readme_words_by_language_dict)
    
    # Make an entry in each dictionary for 'all_languages'
    all_readme_text = " ".join(bootcamp_readme_df.readme_contents)

    readme_words_by_language_dict['all_languages'] = re.sub(r"[^\w\s]", "", all_readme_text).split()

    word_count_by_lanugage_dict['all_languages'] = pd.Series(readme_words_by_language_dict['all_languages']).value_counts()
    
    return readme_words_by_language_dict, word_count_by_lanugage_dict

def generate_word_count_df(word_count_by_lanugage_dict):
    
    return pd.concat(word_count_by_lanugage_dict.values(), axis=1, sort=False)\
    .set_axis(word_count_by_lanugage_dict.keys(), axis=1, inplace=False)\
    .fillna(0).apply(lambda s : s.astype(int))

def get_data_sci_words_df(word_counts_df):
    not_js_in_python = word_counts_df[word_counts_df.JavaScript == 0].sort_values(by='Python').tail(10)
    not_js_in_jupyter = word_counts_df[word_counts_df.JavaScript == 0].sort_values(by='Jupyter Notebook').tail(10)

    not_java_in_python = word_counts_df[word_counts_df.Java == 0].sort_values(by='Python').tail(10)
    not_java_in_jupyter = word_counts_df[word_counts_df.Java == 0].sort_values(by='Jupyter Notebook').tail(10)

    return pd.concat([not_java_in_jupyter, not_java_in_python, not_js_in_jupyter,\
                               not_js_in_python])

def get_web_dev_words_df(word_counts_df):
    not_python_in_js = word_counts_df[word_counts_df.Python == 0].sort_values(by='JavaScript').tail(10)
    not_jupyter_in_js = word_counts_df[word_counts_df['Jupyter Notebook'] == 0].sort_values(by='JavaScript').tail(10)

    not_python_in_java = word_counts_df[word_counts_df.Python == 0].sort_values(by='Java').tail(10)
    not_jupyter_in_java = word_counts_df[word_counts_df['Jupyter Notebook'] == 0].sort_values(by='Java').tail(10)

    return pd.concat([not_jupyter_in_java, not_jupyter_in_js, not_python_in_java, not_python_in_js])

def count_words_between_webdev_datasci(df):
    word_counts_df = df.copy()
    
    word_counts_df['web_dev_languages'] = word_counts_df.Java + word_counts_df.JavaScript
    word_counts_df['data_sci_languages'] = word_counts_df['Jupyter Notebook'] + word_counts_df['Python']
    
    return word_counts_df

def plot_proportion_of_words(word_counts_df):
    (word_counts_df.assign(p_web_dev=word_counts_df.web_dev_languages / word_counts_df.all_languages, \
                      p_data_sci=word_counts_df.data_sci_languages / word_counts_df.all_languages)\
    .sort_values(by='all_languages')\
    [['p_web_dev', 'p_data_sci']]\
    .tail(20).sort_values('p_web_dev').plot.barh(stacked=True))

    plt.title('Proportion of Web Dev and Data Sci for 20 most common words')
    plt.legend(loc="best")
    plt.show()
    
def generate_wordcloud(list_of_words, colormap_value=None, background_color=None, fig_size=(10, 8)):
    img = WordCloud(background_color=background_color, height=1000, width=800, colormap=colormap_value)\
    .generate(" ".join(list_of_words))

    plt.figure(figsize=fig_size)
    plt.imshow(img)
    plt.axis("off")