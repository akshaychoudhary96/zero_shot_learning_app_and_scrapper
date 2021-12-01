import pandas as pd
import streamlit as st
from transformers import pipeline,AutoTokenizer, AutoModelForSequenceClassification
import plotly.express as px
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np

from pygments import highlight
from pygments.lexers import JsonLexer
from pygments.formatters import TerminalFormatter

from google_play_scraper import Sort, reviews, app


@st.cache(allow_output_mutation=True)
def load_model():
    return pipeline("zero-shot-classification")


def classify(sequences: list, candidate_labels: list, multi_label=False):
    """
    Get the classification output

    Args:
        sequences: list - list of sequences you want to classify
        candidate_labels: list - Possible classfication labels
        multi_label: bool - binary or multiple labels

    Returns:
        res: list - list of predicted probabilities for each example
    """
    res = load_model()(sequences, candidate_labels, multi_label=multi_label)

    if isinstance(res, dict):
        return [res]

    return res


def output(res, multi_label):
    """
    Args:
        res: result from classifier
        multi_label: is it multilabel classification
    Returns:
        pd.DataFrame: Pandas DataFrames
    """

    sequences = []
    labels = []
    scores = []

    if multi_label:
        results = []
        for i in range(len(res)):
            record = {
                "example": res[i].get('sequence')
            }
            record.update(zip(res[i].get('labels'), res[i].get('scores')))
            results.append(record)
            print(results)
        return pd.DataFrame.from_records(results)
    else:

        for item in res:
            sequences.append(item.get("sequence"))
            labels.append(item.get('labels')[0])
            scores.append(item.get('scores')[0])

        return pd.DataFrame.from_dict(
            {
                "examples": sequences,
                "labels": labels,
                "scores": scores
            }
        )

def scrapper(app_identifier, country_code):
    app_packages = []
    app_packages.append(app_identifier)
    app_reviews = []
    temp = []
    # app_info = app(app_packages,lang="en",country=country_code)
    my_bar = st.progress(0)
    for score in list(range(1, 6)):
        for sort_order in [Sort.MOST_RELEVANT, Sort.NEWEST]:
            rvs, _ = reviews(
                app_packages[0],
                lang='en',
                country=str(country_code),
                sort=sort_order,
                count= 10,
                filter_score_with=score
            )
        for r in rvs:
            r['sortOrder'] = 'most_relevant' if sort_order == Sort.MOST_RELEVANT else 'newest'
            r['appId'] = app_packages[0]
            temp.append(r['content'])
        my_bar.progress(100-int(100/score))   
        app_reviews.extend(temp)
    # app_reviews_df = pd.DataFrame(app_reviews)
    return app_reviews

    
def run():
    """
    Run the application
    """
    #image = Image.open('/Users/akshaychoudhary/Documents/Carleton University/3rd Term/Advance ML /Assignment 3/zero-shot-classifier-app-master/zero-shot.png')

    #st.sidebar.image(image)
    st.sidebar.info("You can use this app this app to do any simple classification without training")
    st.title("Zero-shot classification App")
    st.text_input("Enter the application name")
    app_identifier = st.text_input("Enter the application identifier it can be found at the end of the url of app in play store")
    country_code = st.text_input("Enter the country code for e.x 'in' for india, 'ca' for canada")
    candidate_labels = st.text_input("Enter the labels separated by comma")
    multi_label = st.checkbox("MultiLabel", [True, False])
    sequences = scrapper(app_identifier,country_code)

    collect_list = lambda x: [str(item) for item in x.split(",")]

    #sequences = st.text_input("Enter the examples separated by comma")
    
    #res = classify(collect_list(sequences), collect_list(candidate_labels), multi_label)
    res = classify(sequences, collect_list(candidate_labels), multi_label)
    data = output(res, multi_label)
    st.dataframe(data)

    if not multi_label:
        fig = px.bar(data, x='examples', y='scores', color='labels')
    else:
        fig = px.bar(data, x='example', y=collect_list(candidate_labels))
    st.plotly_chart(fig)


if __name__ == "__main__":
    run()
