import gradio as gr
from transformers import pipeline
from PIL import Image

# Load pretrained Hugging Face model (example: skin cancer detection)
classifier = pipeline("image-classification", model="VRJBro/skin-cancer-detection")

def predict(image):
    results = classifier(image)
    return {item["label"]: float(item["score"]) for item in results}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="Skin Lesion Detection",
    description="Upload a skin image to detect benign or malignant lesions"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080)
import gradio as gr
from transformers import pipeline
from PIL import Image

# Load pretrained Hugging Face model (example: skin cancer detection)
classifier = pipeline("image-classification", model="VRJBro/skin-cancer-detection")

def predict(image):
    results = classifier(image)
    return {item["label"]: float(item["score"]) for item in results}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="Skin Lesion Detection",
    description="Upload a skin image to detect benign or malignant lesions"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080)
added app.py
