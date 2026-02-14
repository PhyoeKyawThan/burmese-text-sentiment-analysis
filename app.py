import gradio as gr
import numpy as np
from sentiment_engine import BurmeseSentiment

engine = BurmeseSentiment()

def predict_emotion(text):
    return engine.analyze(text)

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Burmese Emotion Tracker")
    gr.Markdown("Enter a Burmese sentence to analyze its sentiment (Positive, Neutral, or Negative).")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Input Text (Burmese)", 
                placeholder="ဒီမှာ စာသားရိုက်ထည့်ပါ...",
                lines=3
            )
            submit_btn = gr.Button("Analyze Sentiment", variant="primary")
        
        with gr.Column():
            output_label = gr.Label(label="Analysis Result")
    
    gr.Examples(
        examples=[
            "ဒီနေ့တော့ တကယ်ကို စိတ်ချမ်းသာဖို့ကောင်းတဲ့ နေ့လေးပါပဲ",
            "မင်းကြောင့် ရင်တွေပူနေတယ်",
            "ရန်ကုန်မြို့တွင် ယနေ့ မိုးရွာသွန်းနိုင်ပါသည်",
            "ဒီဝန်ဆောင်မှုက တကယ်ကို စိတ်ပျက်ဖို့ကောင်းတယ်"
        ],
        inputs=input_text
    )

    submit_btn.click(fn=predict_emotion, inputs=input_text, outputs=output_label)

if __name__ == "__main__":
    demo.launch(debug=True)