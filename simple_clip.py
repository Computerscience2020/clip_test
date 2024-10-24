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
    categories = ["cat", "dog", "horse" , "text"]
    text = clip.tokenize([f"a photo of a {c}" for c in categories]).to(device)
    
    # Get model predictions
    with torch.no_grad():
        image_features = model.encode_image(processed_image)
        text_features = model.encode_text(text)
        
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    
    # Print results
    print("\n🔍 Classification Results:")
    print("------------------------")
    for category, score in zip(categories, similarity[0].tolist()):
        print(f"{category}: {score*100:.2f}%")

# Run the classifier
if __name__ == "__main__":
    # Update this path to match your actual image file
    IMAGE_PATH = "temp_image.jpg"  # If your image is named test.jpg in the same folder
    # Or use the full path, for example:
    # IMAGE_PATH = r"C:\Users\LadyC\OneDrive\Documents\clip_test\your_actual_image.jpg"
    
    print("Starting classification...")
    try:
        simple_classifier(IMAGE_PATH)
    except Exception as e:
        print(f"An error occurred: {e}")
        print("\nTip: Make sure you have an image file in your clip_test folder!")
        print("Currently looking for the image at:", IMAGE_PATH)