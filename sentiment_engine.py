import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from myTokenize import SyllableTokenizer
from myTokenize import WordTokenizer

class BurmeseSentiment:
    def __init__(self, model_path='trained_models/burmese_sentiment_lstm.keras', 
                 tokenizer_path='trained_models/tokenizer.pkl', 
                 encoder_path='trained_models/encoder.pkl'):
        self.model = load_model(model_path)
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        with open(encoder_path, 'rb') as f:
            self.encoder = pickle.load(f)
        # self.syl_tokenizer = WordTokenizer(engine="CRF")
        self.syl_tokenizer = SyllableTokenizer()

    def analyze(self, text):
        syllables = self.syl_tokenizer.tokenize(text)
        seq = self.tokenizer.texts_to_sequences([syllables])
        padded = pad_sequences(seq, maxlen=100)
        
        pred = self.model.predict(padded, verbose=0)[0]
        print(pred)
        results = {self.encoder.classes_[i]: float(pred[i]) for i in range(len(pred))}
        return results
    
if __name__ == "__main__":
    engine = BurmeseSentiment()
    print(engine.analyze("ဒီနေ့တော့ ပျော်စရာကြီး"))
    print(engine.analyze("မင်းကြောင့် ရင်တွေပူနေတယ်"))
    print(engine.analyze("ဒီနေ့တော့ တကယ်ကို စိတ်ချမ်းသာဖို့ကောင်းတဲ့ နေ့လေးပါပဲ"))
    print(engine.analyze("ရန်ကုန်မြို့တွင် ယနေ့ မိုးရွာသွန်းနိုင်ပါသည်"))
