import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline



def train_model_from_df(df):
    x = df["task_description"]
    y = df["priority"]

    model : Pipeline = Pipeline([
        ("vectorizer", CountVectorizer()),
        ("classifier", MultinomialNB())
    ])

    model.fit(x, y)
    joblib.dump(model, "model.pkl")
    return model