import gradio as gr
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from myTokenize import SyllableTokenizer

model = load_model('trained_models/burmese_sentiment_lstm.keras')
with open('tokenizer.pkl', 'rb') as f:
    keras_tokenizer = pickle.load(f)
with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

syl_tokenizer = SyllableTokenizer()

def predict_emotion(text):
    if not text or text.strip() == "":
        return "Please enter some text."
    
    syllables = syl_tokenizer.tokenize(text)
    seq = keras_tokenizer.texts_to_sequences([syllables])
    padded = pad_sequences(seq, maxlen=100)
    
    pred = model.predict(padded, verbose=0)[0]
    
    results = {encoder.classes_[i]: float(pred[i]) for i in range(len(pred))}
    return results

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
    demo.launch(share=True)