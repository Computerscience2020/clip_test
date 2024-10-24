import streamlit as st
import torch
import clip
from PIL import Image

def simple_classifier(image_path):
    # Load the CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # Load and preprocess the image
    image = Image.open(image_path)
    processed_image = preprocess(image).unsqueeze(0).to(device)
    
    # Define simple categories
    categories = ["cat", "dog", "horse"]
    text = clip.tokenize([f"a photo of a {c}" for c in categories]).to(device)
    
    # Get model predictions
    with torch.no_grad():
        image_features = model.encode_image(processed_image)
        text_features = model.encode_text(text)
        
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    
    # Return results
    results = {category: score.item() * 100 for category, score in zip(categories, similarity[0])}
    return results

st.title("Simple Image Classifier with CLIP")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    # Save the uploaded image to a temporary file
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Run the classifier
    results = simple_classifier("temp_image.jpg")
    
    st.write("🔍 Classification Results:")
    for category, score in results.items():
        st.write(f"{category}: {score:.2f}%")
