import env
import numpy as np
import pandas as pd
import utilities as utils
import re

from sklearn.metrics import classification_report, accuracy_score

def idf(word, document_series):
    n_occurences = sum([1 for doc in document_series if word in doc])
    
    return np.log(len(document_series) / n_occurences)

def generate_tf_idf_tfidf_dataframe(word_list, document_series):
    word_freq_df = (pd.DataFrame({'raw_count': word_list.value_counts()})\
                    .assign(frequency=lambda df: df.raw_count / df.raw_count.sum())\
                    .assign(augmented_frequency=lambda df: df.frequency / df.frequency.max()))
    
    word_freq_df = word_freq_df.reset_index()
    word_freq_df = word_freq_df.rename(columns={'index' : 'word'})

    word_freq_df['idf'] = word_freq_df.word.apply(idf, document_series=document_series)
    word_freq_df['tf_idf'] = word_freq_df.frequency * word_freq_df.idf
    
    return word_freq_df

def add_features(vectorized_text, vectorizer, df):
    bootcamp_readme_df = df.copy()
    
    X = pd.DataFrame(vectorized_text.todense(), columns=vectorizer.get_feature_names())

    bootcamp_readme_df['is_webdev'] = bootcamp_readme_df.language.apply(lambda lang : True if (lang == 'Java') | (lang == 'JavaScript') else False)

    is_webdev = bootcamp_readme_df.is_webdev
    is_webdev = is_webdev.reset_index()
    is_webdev = is_webdev.drop(columns=['index'])

    return pd.concat([X, is_webdev], axis=1)

def print_model_evaluation(sample_df):
    print('Accuracy: {:.2%}'.format(accuracy_score(sample_df.actual, sample_df.predicted)))
    print('---')
    print('Confusion Matrix')
    print(pd.crosstab(sample_df.predicted, sample_df.actual))
    print('---')
    print(classification_report(sample_df.actual, sample_df.predicted))
    
def predict_language_from_readme(trained_model, vectorizer, readme_txt, is_webdev):
    
    readme_txt = utils.nlp_basic_clean(readme_txt)
    readme_txt = re.sub(r"\n", " ", readme_txt)
    readme_txt = utils.nlp_tokenize(readme_txt)
    readme_txt = utils.nlp_remove_stopwords(readme_txt, extra_words=['de', 'e', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0'])
    readme_txt = utils.nlp_lemmatize(readme_txt)
    
    readme_txt = pd.Series(readme_txt)
    
    vectorized_readme = vectorizer.transform(readme_txt)
    vectorized_readme = pd.DataFrame(vectorized_readme.todense(), columns=vectorizer.get_feature_names())
    vectorized_readme = pd.concat([vectorized_readme, pd.Series(is_webdev)], axis=1)
    
    return trained_model.predict(vectorized_readme)