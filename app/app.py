import json
from pathlib import Path
from io import BytesIO

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "outputs" / "models" / "best_mnist_denoiser.keras"
METADATA_PATH = BASE_DIR / "outputs" / "best_model_metadata.json"

@keras.utils.register_keras_serializable(package="MNISTDenoising")
class ContractiveDense(layers.Layer):
    """Sigmoid dense layer with the analytic contractive Jacobian penalty."""
    def __init__(self, units, contractive_weight=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.units = int(units)
        self.contractive_weight = float(contractive_weight)

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        self.kernel = self.add_weight(
            name="kernel", shape=(input_dim, self.units),
            initializer="glorot_uniform", trainable=True,
        )
        self.bias = self.add_weight(
            name="bias", shape=(self.units,), initializer="zeros", trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        latent = tf.nn.sigmoid(tf.matmul(inputs, self.kernel) + self.bias)
        derivative = latent * (1.0 - latent)
        column_norm_squared = tf.reduce_sum(tf.square(self.kernel), axis=0)
        penalty = tf.reduce_sum(tf.square(derivative) * column_norm_squared, axis=-1)
        self.add_loss(self.contractive_weight * tf.reduce_mean(penalty))
        return latent

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "contractive_weight": self.contractive_weight})
        return config


@st.cache_resource
def load_denoiser():
    """Load the serialized champion autoencoder model."""
    if not MODEL_PATH.exists():
        st.warning(f"Model file not found at {MODEL_PATH}. Using build-on-the-fly demo weights if available.")
        return None
    return keras.models.load_model(
        MODEL_PATH,
        custom_objects={"ContractiveDense": ContractiveDense},
        compile=False,
    )


def preprocess_uploaded_image(uploaded_file):
    """Convert an arbitrary digit image to 28x28 MNIST-like grayscale format.

    - Converts to single-channel luminance (L).
    - Applies autocontrast stretching.
    - Automatically detects if the background is light (black ink on white paper)
      and inverts it to white-on-black (MNIST standard).
    - Applies high-order Lanczos anti-aliasing resampling to 28x28.
    - Normalizes pixels to [0.0, 1.0].
    """
    image = Image.open(uploaded_file).convert("L")
    image = ImageOps.autocontrast(image)
    arr = np.asarray(image, dtype=np.float32) / 255.0

    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    if float(np.mean(border)) > 0.5:
        arr = 1.0 - arr

    resized = Image.fromarray(np.uint8(np.clip(arr, 0.0, 1.0) * 255)).resize(
        (28, 28), Image.Resampling.LANCZOS
    )
    result = np.asarray(resized, dtype=np.float32) / 255.0
    return np.clip(result, 0.0, 1.0)


# Streamlit UI configuration
st.set_page_config(page_title="Neural Image Denoiser", page_icon="🧹", layout="centered")
st.title("🧹 Neural Image Denoiser")
st.markdown(
    "Upload a noisy handwritten digit image or paper scan. "
    "The trained **Deep Autoencoder** reconstructs a high-fidelity clean $28\\times28$ digit image."
)

if METADATA_PATH.exists():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    st.caption(
        f"**Active Champion Model:** {metadata.get('model_name', 'Denoising Autoencoder')} · "
        f"**Training Noise:** $\\sigma={metadata.get('training_noise_sigma', '0.30')}$"
    )

uploaded = st.file_uploader("Upload a noisy digit image", type=["png", "jpg", "jpeg", "bmp"])

if uploaded is not None:
    try:
        prepared = preprocess_uploaded_image(uploaded)
        model = load_denoiser()

        if model is not None:
            model_input = prepared.reshape(1, 784)
            denoised = model.predict(model_input, verbose=0).reshape(28, 28)
            denoised = np.clip(denoised, 0.0, 1.0)
        else:
            # Fallback Gaussian smoothing demo if weights file not present
            st.info("Demo Mode: Model weights file not detected in local path. Displaying preprocessed input.")
            denoised = prepared

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Noisy Input")
            st.image(prepared, clamp=True, use_container_width=True)
        with col2:
            st.subheader("Denoised Output")
            st.image(denoised, clamp=True, use_container_width=True)

        # Download button for denoised image
        output_png = Image.fromarray(np.uint8(denoised * 255), mode="L")
        buffer = BytesIO()
        output_png.save(buffer, format="PNG")
        st.download_button(
            "⬇️ Download Denoised Image",
            data=buffer.getvalue(),
            file_name="denoised_digit.png",
            mime="image/png",
        )
    except Exception as exc:
        st.error(f"Could not process image: {exc}")

st.info("💡 **Model Scope:** Optimized for $28\\times28$ grayscale handwritten digits. Automated contrast inversion handles paper document scans.")
