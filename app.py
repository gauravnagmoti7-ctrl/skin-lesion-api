import os
from flask import Flask, request, jsonify
import gradio as gr
from transformers import pipeline
from PIL import Image

# Initialize Flask
app = Flask(__name__)

# Load Hugging Face pipelines
skin_model = pipeline("image-classification", model="VRJBro/skin-cancer-detection")
eye_model = pipeline("image-classification", model="RetinaNet-ResNet50-FPN")  # placeholder, change if needed
oral_model = pipeline("image-classification", model="microsoft/resnet-50")    # placeholder, change if needed

# Flask API route
@app.route("/")
def home():
    return {"message": "AI Health Scan is running on Render!"}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["file"]
        image = Image.open(file.stream)

        results = {
            "skin": skin_model(image)[0],
            "eye": eye_model(image)[0],
            "oral": oral_model(image)[0]
        }
        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)})

# Gradio UI
def classify(image):
    return {
        "Skin": skin_model(image),
        "Eye": eye_model(image),
        "Oral": oral_model(image)
    }

demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil"),
    outputs=["label", "label", "label"],
    title="AI Health Scan",
    description="Upload an image to detect skin, eye, or oral issues."
)

@app.route("/gradio")
def gradio_app():
    return demo.launch(share=False, inline=True)

# Entry point for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
