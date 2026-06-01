import json
from io import BytesIO
from typing import Any

import numpy as np
import requests
import streamlit as st
from PIL import Image


API_DEFAULT_URL = "http://127.0.0.1:8000/predict"
EXPECTED_FEATURE_COUNT = 28 * 28 * 3
LABEL_NAME_MAP = {
    "0": "nv",
    "1": "bkl",
    "2": "mel/bcc",
}


def load_image(image_bytes: BytesIO) -> Image.Image:
    """Charge une image depuis un flux et la convertit en RGB."""
    image = Image.open(image_bytes)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def preprocess_image(image: Image.Image, size: tuple[int, int] = (28, 28)) -> np.ndarray:
    """Redimensionne l'image, la convertit en RGB, et aplatit en vecteur 2352."""
    image_resized = image.resize(size, resample=Image.LANCZOS)
    array = np.asarray(image_resized, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("L'image doit être en couleurs RGB avec 3 canaux.")
    flattened = array.reshape(-1)
    if flattened.shape[0] != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Le vecteur d'entrée doit contenir {EXPECTED_FEATURE_COUNT} valeurs, "
            f"mais en contient {flattened.shape[0]}.")
    return flattened


def post_predict(api_url: str, pixels: list[float], normalize: bool = True) -> dict[str, Any]:
    """Envoie une requête POST à l'API FastAPI et retourne la réponse JSON."""
    payload = {
        "pixels": pixels,
        "normalize": normalize,
    }
    response = requests.post(api_url, json=payload, timeout=15)
    response.raise_for_status()
    return response.json()


def format_probabilities(probs: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(probs.items(), key=lambda item: item[0])


def get_label_name(label: str | int) -> str:
    raw_label = str(label)
    return LABEL_NAME_MAP.get(raw_label, raw_label)


def format_label_display(label: str | int) -> str:
    raw_label = str(label)
    label_name = get_label_name(raw_label)
    return f"{label_name} ({raw_label})"


def main() -> None:
    st.set_page_config(
        page_title="Classification de lésions cutanées",
        page_icon="🩺",
        layout="centered",
    )

    st.title("Classification de lésions cutanées")
    st.markdown(
        """
        Téléversez une image JPG/PNG de lésion cutanée. L'image sera redimensionnée en 28×28, convertie en RGB,
        aplatie en un vecteur de 2352 valeurs, puis envoyée à l'API de prédiction.
        """
    )

    api_url = st.text_input("URL de l'API de prédiction", API_DEFAULT_URL)
    uploaded_file = st.file_uploader("Téléverser une image", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        st.info("Téléversez une image pour lancer la prédiction.")
        return

    try:
        image = load_image(uploaded_file)
    except Exception as exc:
        st.error(f"Impossible de charger l'image : {exc}")
        return

    st.image(image, caption="Image téléchargée", use_column_width=True)
    st.write(f"Taille originale : {image.width} × {image.height}")

    try:
        processed_vector = preprocess_image(image)
    except Exception as exc:
        st.error(f"Erreur de prétraitement : {exc}")
        return

    st.success("Image correctement prétraitée.")
    st.write(f"Vecteur aplati : {processed_vector.shape[0]} valeurs")

    st.markdown("---")
    st.subheader("Résumé de l'entrée")
    st.write(
        "L'image est convertie en RGB, redimensionnée à 28x28, puis aplatie en un vecteur 1D de 2352 valeurs."
    )
    st.write(f"Nombre de features attendu : {EXPECTED_FEATURE_COUNT}")
    st.write("Normalize : True")

    if st.button("Envoyer à l'API de prédiction"):
        with st.spinner("Appel de l'API de prédiction..."):
            try:
                result = post_predict(api_url, processed_vector.tolist(), normalize=True)
            except requests.exceptions.RequestException as exc:
                st.error(f"Erreur réseau ou API : {exc}")
                return
            except ValueError as exc:
                st.error(f"ERREUR de parsing de la réponse : {exc}")
                return

        st.success("Prédiction reçue")
        st.write("### Résultat")
        predicted_index = result.get('predicted_index', 'N/A')
        predicted_label = result.get('predicted_label', 'N/A')
        st.write(f"**Label prédit** : {format_label_display(predicted_label)}")
        st.write(f"**Latence** : {result.get('latency_ms', 'N/A'):.2f} ms")

        probs = result.get("probabilities", {})
        if isinstance(probs, dict):
            st.write("### Probabilités par classe")
            for label, score in format_probabilities(probs):
                st.write(f"- {format_label_display(label)} : {score:.4f}")
        else:
            st.warning("Aucune probabilité disponible dans la réponse.")

        st.write("### Payload envoyé à l'API")
        st.json({
            "pixels_length": len(processed_vector),
            "normalize": True,
            "api_url": api_url,
        })


if __name__ == "__main__":
    main()
