import joblib
import numpy as np
from flask import Flask, request, render_template
from preprocess import preprocess  # From your existing file
from features import generate_all_features

# Load model and vectorizer using joblib
try:
    with open("model.pkl", "rb") as f:
        model = joblib.load(f)
    
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = joblib.load(f)
    print("✅ Models and vectorizer loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    model = None
    vectorizer = None

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if not model or not vectorizer:
        return render_template("index.html", 
                               prediction_text="Error: Model not loaded. Please check your model files.")

    q1 = request.form.get("question1", "")
    q2 = request.form.get("question2", "")

    # Preprocess questions using your existing preprocess function
    q1_clean = preprocess(q1)
    q2_clean = preprocess(q2)

    # Generate all hand-crafted features
    manual_features = generate_all_features(q1_clean, q2_clean)
    
    # Vectorize the questions separately, as you did during training
    # This ensures the output has the correct shape for concatenation
    q1_vectorized = vectorizer.transform([q1_clean])
    q2_vectorized = vectorizer.transform([q2_clean])

    # Concatenate all features into a single NumPy array
    combined_features = np.hstack([np.array([manual_features]), q1_vectorized.toarray(), q2_vectorized.toarray()])

    # Print the final features shape to verify it matches what the model expects (6022 features)
    print(f"Final input features shape: {combined_features.shape}")

    # Predict
    prediction = model.predict(combined_features)[0]
    result = "Duplicate" if prediction == 1 else "Not Duplicate"

    return render_template("index.html", 
                           prediction_text=f"Result: {result}",
                           q1=q1, q2=q2)

if __name__ == "__main__":
    app.run(debug=True)