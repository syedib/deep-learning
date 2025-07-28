# import necessary libraries
import pandas as pd
import spacy

# download the spacy model
nlp = spacy.load("en_core_web_sm")

# helper functions from text preprocessing section
def lower_replace(series):
    output = series.str.lower()
    output = output.str.replace(r'\[.*?\]', '', regex=True)
    output = output.str.replace(r'[^\w\s]', '', regex=True)
    return output

def token_lemma_nonstop(text):
    doc = nlp(text)
    output = [token.lemma_ for token in doc if not token.is_stop]
    return ' '.join(output)

def clean_and_normalize(series):
    output = lower_replace(series)
    output = output.apply(token_lemma_nonstop)
    return output

# allow command-line execution
if __name__ == "__main__":
    print("Text preprocessing module ready to use.")