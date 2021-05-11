# Whipping the Model into Shape with Bootcamps!

## Goal
This project used the text from README files in coding bootcamp repositories on Github to predict the main coding language present in the repository.

## Setup this project
* Dependencies
    1. [utilities.py](https://github.com/david-ryan-alviola/utilities/releases)
        * Follow the instructions to use the latest features
    2. python
    3. pandas
    4. scipy
    5. sklearn
    6. numpy
    7. matplotlib.pyplot
    8. seaborn
    9. wordcloud
    10. requests
* Steps to recreate
    1. Clone this repository
    2. Install `utilities.py` according to the instructions
    3. Setup env.py
        * Remove the .template extension (should result in `env.py`)
        * Fill in your user_name, password, host, and data_path
    4. Open `coding_bootcamp_language_prediction.ipynb` and run the cells
        * Follow the instructions in the notebook for acquiring the data from GitHub if it's the first time running the notebook

## Key Findings
1. Model had 81% overall accuracy on test sample data
    * High precision and recall for JavaScript (84%, 100%) and Jupyter Notebook (87%, 81%)
2. "Python" had the second highest TF across all README files and the highest TF-IDF
3. Still have room for improvement
    * Model struggles to identify Java repositories (0% for both precision and recall on test data)
    * Model has lower precision and recall for Python than Jupyter Notebook (65%, 73%)

## The plan
I wanted to create a classification model that would predict the language of coding bootcamp repositories. I planned to use the GitHub API to retrieve 1000 of the most starred repositories with "bootcamp" in their name. After my first iteration, I decided to focus on Java and JavaScript (web development languages) and Jupyter Notebook and Python (data science languages). This helped me narrow the scope of my project and improved the performance of the model. I also decided to add the `is_webdev` feature to the model which helped improve model performance further.

## Data Dictionary
This is the structure of the data for the second model:
#### Target
Name | Description | Type
:---: | :---: | :---:
language | The main programming language of the repository | string
#### Features
Name | Description | Type
:---: | :---: | :---:
repo | The name of the repository | string
readme_contents | The full text of the repositoriy's README file | string
is_webdev | Indicates if the repository is Java or JavaScript | string

## Results
Highly accurate model (81% overall) with excellent precision and recall for JavaScript and Jupyter Notebook.

## Recommendations
1. Add more Java and Python observations to better train the model on discerning those repositories from JavaScript and Jupyter Notebook
    * Preferably enough to have equal representation of all languages
1. Split the model between data science and web development languages
    * Remove the need to add `is_webdev` feature