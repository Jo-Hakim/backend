from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline, Conversation

app = Flask(__name__)
CORS(app)

# Load conversational model
bobert = pipeline("conversational", model="microsoft/DialoGPT-small")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True)
    if not data or "message" not in data:
        return jsonify({"error": "No message provided"}), 400

    user_input = data["message"]

    # Inject Bobert personality (optional)
    if "your name" in user_input.lower():
        return jsonify({"response": "My name is Bobert 😎"})

    # Generate conversation
    conv = Conversation(user_input)
    result = bobert(conv)

    return jsonify({"response": result.generated_responses[-1]})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=3000)
