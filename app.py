import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from models import Transformer  
import matplotlib.pyplot as plt
from types import SimpleNamespace
from configs import run_config
import data
from math import sqrt

# Load trained model
config = SimpleNamespace(**run_config)

train_loader, test_loader = data.get_mnist_data(batch_size=config.batch_size)

# Initialise the model
model = Transformer(
    num_patches=config.num_patches,
    patch_dim=int(28 / sqrt(config.num_patches)),
    embedding_size=config.embedding_size,
    num_layers=config.num_layers)

model.load_state_dict(torch.load("model_weights.pth"))
model.eval()

# Define transform (match training)
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# UI
st.title("🖌️ Handwritten Digit Classifier")
st.write("Draw a digit (0–9) below and submit it for classification.")

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("🧠 Predict"):
    if canvas_result.image_data is not None:
        img = Image.fromarray(np.uint8(canvas_result.image_data)).convert("L")
        input_tensor = transform(img).unsqueeze(0)
        print("Input tensor shape:", input_tensor)

        fig, ax = plt.subplots()
        ax.imshow(input_tensor.squeeze().numpy(), cmap='gray')
        ax.axis('off')
        st.pyplot(fig)          

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs).item()
            confidence = probs[0][pred].item()

        st.success(f"**Prediction:** {pred}")
        st.info(f"**Confidence:** {confidence:.2%}")

        true_label = st.text_input("True Label (optional, for feedback):", max_chars=1)
        if true_label:
            st.write(f"✅ Thanks! You entered: **{true_label}**")
    else:
        st.warning("Please draw a digit before submitting.")
