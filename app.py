from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load("model/hate_speech_model.pkl")
tfidf_vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

# Define the home route to serve index.html
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get the input data from JSON
    text = data['text']
    
    # Preprocess the text using TF-IDF vectorizer
    text_tfidf = tfidf_vectorizer.transform([text])
    
    # Predict using the model
    prediction = model.predict(text_tfidf)
    
    # Map prediction to labels
    if prediction == 0:
        result = "Offensive Language"
    else:
        result = "Neutral"
    
    # Return prediction as JSON
    return jsonify({"prediction": result})

if __name__ == '__main__':
    app.run(debug=True)
