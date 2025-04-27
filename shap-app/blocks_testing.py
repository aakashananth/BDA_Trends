import gradio as gr
#import transformers
import shap
import matplotlib.pyplot as plt
import pickle
import io
from PIL import Image

#dataset = datasets.load_dataset("imdb", split="test")
#short_data = [v[:500] for v in dataset["text"][:100]]
#classifier = transformers.pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
#explainer = shap.Explainer(classifier)
#shap_values = explainer(short_data)


# Then load it next time
with open("shap_values.pkl", "rb") as f:
    shap_values = pickle.load(f)


def n_bar_plot_image(s_values):
    fig = plt.figure()
    shap.plots.bar(s_values[:, :, "POSITIVE"].mean(0),order=shap.Explanation.argsort, show=False)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    img = Image.open(buf)
    return img

def p_bar_plot_image(s_values):
    fig = plt.figure()
    shap.plots.bar(s_values[:, :, "POSITIVE"], show=False)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    img = Image.open(buf)
    return img


def review_analysis(review):
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    explainer = shap.Explainer(model)
    rev_sv = explainer([review])
    tp = shap.plots.text(rev_sv[:,:,'POSITIVE'],display=False)
    return f"<head>{shap.getjs()}</head><body>{tp}</body>"


with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown(
        """
        # SHAP VALUE DASHBOARD
        ## What Are SHAP Values?
        ### SHapley Additive exPlanations or SHAP for short is is a method to explain the output of machine learning models by quantifying the contribution of each feature to the prediction. SHAP values are derived from game theory and provide a fair distribution of contributions from each input feature
        ## How SHAP values work for Sentiment Analysis
        ### SHAP values reveal how much each word in the text contributes to the predicted sentiment either positive or negative. Each word in the input text receives a SHAP value that indicates its impact on the model’s sentiment prediction.
        ## What is the difference between Global and Local Explanation?
        ### *Global*: Aggregate SHAP values across multiple instances to identify the most influential words in sentiment analysis.
        
        ### *Local*: For individual text samples, SHAP values explain how specific words influenced the sentiment prediction for that sample.
        """)
    with gr.Row():
        gr.Markdown("""## GLOBAL IMPACT""")
    with gr.Row():
        with gr.Column():
            gr.Markdown("""### These are the top words for driving positive sentiment across hundreds of positive reviews 
                         Red Contributed to a positive sentiment
                        """)
            gr.Image(value=p_bar_plot_image(shap_values), type="pil")
        with gr.Column():
            gr.Markdown("""### These are the top words for driving negative sentiment across hundreds of negative reviews 
                         Blue Contributed to a negative sentiment
                        """)
            gr.Image(value=n_bar_plot_image(shap_values), type="pil")
    with gr.Row():
        gr.Markdown("""## LOCAL IMPACT""")
    with gr.Row():
        gr.Markdown("""### This will showcase how each word in your review contributes to the review's predicted sentiment""")
    with gr.Row():
        inp = gr.Textbox(placeholder="Put your review here", label="Share one of your recent reviews! Please type a decent sized review to get better results.")
    btn = gr.Button("Submit Review")
    with gr.Row():
        gr.Markdown("""### Words on the left side are postively impacting the sentiment prediction score and words on the right side are negatively impacting the prediction score""")
    with gr.Row():
        out = gr.HTML()
    btn.click(fn=review_analysis,inputs=inp,outputs=out)
demo.launch()
