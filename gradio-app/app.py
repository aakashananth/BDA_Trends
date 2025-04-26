import gradio as gr
import pandas as pd
import plotly.express as px
import io
from azure.storage.blob import BlobServiceClient

# Azure Blob Config
storage_account_name = "" # Replace with your Azure Storage Account Name
storage_account_key = "" # Replace with your Azure Storage Account Key
container_name = "" # Replace with your Azure Blob Container Name
connection_string = f"DefaultEndpointsProtocol=https;AccountName={storage_account_name};AccountKey={storage_account_key};EndpointSuffix=core.windows.net"

# Load CSV from Azure Blob
def load_csv_from_blob(blob_name):
    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)
        blob_client = container_client.get_blob_client(blob_name)
        blob_data = blob_client.download_blob().readall()
        return pd.read_csv(io.BytesIO(blob_data))
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})

# KPI Generator
def get_overview(model_name):
    df = load_csv_from_blob("sentiment_result.csv")
    if model_name not in df.columns:
        return "Invalid model", "", "", ""
    total = len(df)
    counts = df[model_name].value_counts()
    pos = counts.get("POSITIVE", 0) / total * 100
    neu = counts.get("NEUTRAL", 0) / total * 100
    neg = counts.get("NEGATIVE", 0) / total * 100
    return f"{total} reviews", f"✅ Positive: {pos:.2f}%", f"😐 Neutral: {neu:.2f}%", f"❌ Negative: {neg:.2f}%"

# Sentiment Breakdown
def plot_sentiment_breakdown():
    df = load_csv_from_blob("sentiment_result.csv")
    models = ["gemini_sentiment", "huggingface_sentiment", "vader_sentiment", "azure_sentiment"]
    summary = []
    for m in models:
        counts = df[m].value_counts(normalize=True) * 100
        summary.append({
            "Model": m.replace("_sentiment", "").capitalize(),
            "Positive": counts.get("POSITIVE", 0),
            "Neutral": counts.get("NEUTRAL", 0),
            "Negative": counts.get("NEGATIVE", 0)
        })
    melt_df = pd.DataFrame(summary).melt(id_vars="Model", var_name="Sentiment", value_name="Percentage")
    fig = px.bar(melt_df, x="Model", y="Percentage", color="Sentiment", barmode="group")
    return fig

# Sentiment Over Time
def plot_sentiment_over_time(model_name, start_date, end_date):
    df = load_csv_from_blob("sentiment_result.csv")
    if model_name not in df.columns or "date" not in df.columns:
        return gr.update(value=None, visible=False)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    try:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        df = df[(df["date"] >= start) & (df["date"] <= end)]
    except Exception:
        return gr.update(value=None, visible=False)

    agg = (
        df.groupby([df["date"].dt.to_period("M"), model_name])
        .size()
        .reset_index(name="count")
    )
    agg["date"] = agg["date"].dt.to_timestamp()
    total = agg.groupby("date")["count"].transform("sum")
    agg["percentage"] = agg["count"] / total * 100

    fig = px.line(
        agg,
        x="date",
        y="percentage",
        color=model_name,
        title=f"Sentiment Over Time - {model_name}",
        labels={model_name: "Sentiment"}
    )
    return fig

# Gemini Summary Viewer
def show_gemini_summary():
    df = load_csv_from_blob("gemini_sentiment_summary.csv")
    return "\n\n".join(df.iloc[:, 0].values.tolist())

# Filtered review preview
def filter_reviews(model_name, sentiment):
    df = load_csv_from_blob("sentiment_result.csv")
    if model_name in df.columns:
        filtered = df[df[model_name] == sentiment].copy()
    else:
        filtered = df.copy()
    return filtered.head(50)

# Build Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 📊 Sentiment Analysis Dashboard")

    with gr.Row():
        model_choice = gr.Dropdown(
            choices=["gemini_sentiment", "huggingface_sentiment", "vader_sentiment", "azure_sentiment"],
            value="gemini_sentiment",
            label="Select Model"
        )

    with gr.Row():
        total_out = gr.Textbox(label="Total Reviews")
        pos_out = gr.Textbox(label="Positive %")
        neu_out = gr.Textbox(label="Neutral %")
        neg_out = gr.Textbox(label="Negative %")

    model_choice.change(fn=get_overview, inputs=model_choice, outputs=[total_out, pos_out, neu_out, neg_out])
    demo.load(fn=get_overview, inputs=[model_choice], outputs=[total_out, pos_out, neu_out, neg_out])

    gr.Markdown("### 📊 Sentiment Breakdown by Model")
    sentiment_chart = gr.Plot()
    demo.load(fn=plot_sentiment_breakdown, inputs=[], outputs=sentiment_chart)

    gr.Markdown("### 📈 Sentiment Over Time")
    with gr.Row():
        time_model = gr.Dropdown(
            choices=["gemini_sentiment", "huggingface_sentiment", "vader_sentiment", "azure_sentiment"],
            value="gemini_sentiment",
            label="Select Model"
        )
        start_date = gr.Textbox(label="Start Date (YYYY-MM-DD)", value="2017-01-01")
        end_date = gr.Textbox(label="End Date (YYYY-MM-DD)", value="2018-01-01")

    time_plot = gr.Plot()
    gr.Button("Show Trend").click(fn=plot_sentiment_over_time, inputs=[time_model, start_date, end_date], outputs=time_plot)
    demo.load(fn=plot_sentiment_over_time, inputs=[time_model, start_date, end_date], outputs=[time_plot])

    gr.Markdown("### 🧠 Gemini Summary Insights")
    summary_box = gr.Textbox(label="Summary", lines=6)
    demo.load(fn=show_gemini_summary, inputs=[], outputs=summary_box)

    gr.Markdown("### 🔍 Explore Reviews")
    with gr.Row():
        review_model = gr.Dropdown(choices=["gemini_sentiment", "huggingface_sentiment", "vader_sentiment", "azure_sentiment"], value="gemini_sentiment")
        sentiment_filter = gr.Dropdown(choices=["POSITIVE", "NEUTRAL", "NEGATIVE"], value="POSITIVE")

    review_table = gr.Dataframe()
    gr.Button("Filter Reviews").click(fn=filter_reviews, inputs=[review_model, sentiment_filter], outputs=review_table)
    demo.load(fn=filter_reviews, inputs=[review_model, sentiment_filter], outputs=[review_table])  # Load table on app start

demo.launch(share=True)
