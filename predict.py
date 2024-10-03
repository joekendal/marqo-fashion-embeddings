import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
from cog import BasePredictor, Input, Path
import requests
from typing import List
from io import BytesIO


class Predictor(BasePredictor):

    def setup(self):
        # Load the marqo-fashionSigLIP model and processor from Hugging Face
        self.model = AutoModel.from_pretrained('Marqo/marqo-fashionSigLIP', trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained('Marqo/marqo-fashionSigLIP', trust_remote_code=True)

    def predict(self, 
        images: List[Path] = Input(description="The product image(s)"), 
        text: str = Input(description="The relevant product text attributes joined by commas"),
        combined: bool = Input(default=True, description="Whether to return the combined image and text embeddings instead of separate ones")
    ) -> dict:
        # Split text input into a list
        text = text.split(",")

        # Download the images from the URLs and convert them to PIL Images
        processed_images = []
        for image in images:
            try:
                if(image.startswith('http')):
                    response = requests.get(image)
                    img = Image.open(BytesIO(response.content))
                else:
                    img = Image.open(image)
                                    
                processed_images.append(img)
            except Exception as e:
                raise ValueError(f"Error loading image: {e}")            


        # Preprocess the image and text
        processed = self.processor(text=text, images=processed_images, padding='max_length', return_tensors="pt")

        # Extract image and text embeddings without gradients (for inference)
        with torch.no_grad():
            image_embeddings = self.model.get_image_features(processed['pixel_values'], normalize=True)
            text_embeddings = self.model.get_text_features(processed['input_ids'], normalize=True)

        # Concatenate image and text embeddings for a combined vector
        combined_embedding = torch.cat([image_embeddings, text_embeddings], dim=-1)

        if not combined:
            # Convert embeddings to list format for JSON response
            image_embedding_list = image_embeddings.cpu().numpy().tolist()
            text_embedding_list = text_embeddings.cpu().numpy().tolist()
            # Return the embeddings separately
            return {
                "image_embedding": image_embedding_list,
                "text_embedding": text_embedding_list,
            }

        combined_embedding_list = combined_embedding.cpu().numpy().tolist()
        # Return the combined one
        return {
            "combined_embedding": combined_embedding_list
        }
