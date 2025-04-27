import transformers
import shap
import datasets
import pickle

# dataset = datasets.load_dataset("imdb", split="test")
# short_data = [v[:500] for v in dataset["text"][:100]]
# classifier = transformers.pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
# explainer = shap.Explainer(classifier)
# shap_values = explainer(short_data)

# dataset = datasets.load_dataset("imdb", split="test")
# bg_data = [v[:500] for v in dataset["text"][100:111]]


# with open("model.pkl", "wb") as f:
#     pickle.dump(explainer, f)

# with open("model.pkl", "wb") as f:
#     pickle.dump(classifier, f)

# with open("shap_values.pkl", "wb") as f:
#     pickle.dump(shap_values, f)

# with open("bg_data.pkl", "wb") as f:
#     pickle.dump(bg_data, f)