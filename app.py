from flask import Flask, render_template, request
from textblob import TextBlob

app = Flask(__name__, static_folder='static', static_url_path='/static')


def predict_sentiment(sentence):
    analysis = TextBlob(sentence)
    if analysis.sentiment.polarity > 0:
        return "positive"
    else:
        return "negative"


@app.route("/", methods=["POST", "GET"])
def index():
    if request.method == "POST":
        input_sentence = request.form["input_sentence"]
        sentiment = predict_sentiment(input_sentence)
        return render_template("index.html", sentiment=sentiment, input_sentence=input_sentence)
    return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=True)
