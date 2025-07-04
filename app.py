import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from models import Transformer  
from types import SimpleNamespace
from configs import hyperparameters
from math import sqrt
from database import Database

# Initialise database connection
database = Database()

# Load trained model
config = SimpleNamespace(**hyperparameters)

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
    stroke_width=25,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Initialize session state
if "predicted" not in st.session_state:
    st.session_state.predicted = False
if "pred" not in st.session_state:
    st.session_state.pred = None
if "confidence" not in st.session_state:
    st.session_state.confidence = None
if "input_tensor" not in st.session_state:
    st.session_state.input_tensor = None

input_tensor = None

# Button and prediction logic
if st.button("🧠 Predict"):
    if canvas_result.image_data is not None:
        img = Image.fromarray(np.uint8(canvas_result.image_data)).convert("L")
        input_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs).item()
            confidence = probs[0][pred].item()

        st.session_state.pred = pred
        st.session_state.confidence = confidence
        st.session_state.predicted = True
        st.session_state.input_tensor = input_tensor.squeeze(0).squeeze(0)
    else:
        st.warning("Please draw a digit before submitting.")

pred_placeholder = st.empty()
conf_placeholder = st.empty()
input_placeholder = st.empty()

if st.session_state.predicted:
    pred_placeholder.success(f"**Prediction:** {st.session_state.pred}")
    conf_placeholder.info(f"**Confidence:** {st.session_state.confidence:.2%}")

    true_label = input_placeholder.text_input("True Label:", max_chars=1)
    if true_label:
        # Save data to database
        database.save_row(st.session_state.input_tensor, st.session_state.pred, true_label)

        st.write(f"✅ Thanks! You entered: **{true_label}**")
        st.session_state.predicted = False
        pred_placeholder.empty()
        conf_placeholder.empty()
        input_placeholder.empty()