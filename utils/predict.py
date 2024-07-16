import pickle
from utils.transformer import SentimentDetectorPreprocessor

MODEL_PATH = r"C:\Users\csyas\OneDrive\Desktop\projects\sentiment-analysis\model.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


def get_predictions(text):
    results = {}
    for word in text.split():
        prediction = model.predict_proba(word)[0]
        results[word] = prediction
    all_prediction = model.predict_proba(text)[0]
    results[text] = all_prediction
    return results
