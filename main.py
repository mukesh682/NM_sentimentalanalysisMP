import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from transformers import pipeline
from datetime import datetime
import logging

logging.basicConfig(filename='sentiment_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

class SentimentAnalyzer:
    def __init__(self):
        model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
        self.analyzer = pipeline("sentiment-analysis", model=model_name)
        self.session_id = datetime.now().strftime("%Y%m%d%H%M%S")
        self.results = []

    def analyze(self, text):
        if not text.strip():
            return {"label": "INVALID", "score": 0.0}
        result = self.analyzer(text)[0]
        entry = {
            "timestamp": datetime.now().isoformat(),
            "text": text,
            "label": result["label"],
            "score": result["score"]
        }
        self.results.append(entry)
        logging.info(f"{text} --> {result['label']} ({result['score']:.4f})")
        return result

    def show(self, result, text):
        print("\n========= Sentiment Analysis =========")
        print(f"Text       : {text}")
        print(f"Sentiment  : {result['label']}")
        print(f"Confidence : {result['score']:.4f}")
        print("======================================")

    def save(self):
        if not self.results:
            return
        with open("sentiment_results.txt", "a", encoding="utf-8") as f:
            f.write(f"\n--- Session {self.session_id} ---\n")
            for r in self.results:
                f.write(f"{r['timestamp']} | {r['text']} | {r['label']} | {r['score']:.4f}\n")
            f.write(f"--- End of Session {self.session_id} ---\n")

def main():
    analyzer = SentimentAnalyzer()
    print("Sentiment Analysis System (type 'exit' to quit)\n")
    while True:
        text = input("Enter a sentence: ")
        if text.lower() == "exit":
            analyzer.save()
            print("Session saved. Exiting.")
            break
        result = analyzer.analyze(text)
        analyzer.show(result, text)

if __name__ == "__main__":
    main()
