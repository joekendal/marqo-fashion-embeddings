import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self):
        # Load the marqo-fashionSigLIP model and processor from Hugging Face
        self.model = AutoModel.from_pretrained('Marqo/marqo-fashionSigLIP', trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained('Marqo/marqo-fashionSigLIP', trust_remote_code=True)

    def predict(self, 
        image: Path = Input(description="Input image"), 
        text: str = Input(description="Comma-separated list of text descriptions")
    ) -> dict:
        # Split text input into a list
        text = text.split(",")

        # Preprocess the image and text
        processed = self.processor(text=text, images=[image], padding='max_length', return_tensors="pt")

        # Extract image and text embeddings without gradients (for inference)
        with torch.no_grad():
            image_embeddings = self.model.get_image_features(processed['pixel_values'], normalize=True)
            text_embeddings = self.model.get_text_features(processed['input_ids'], normalize=True)

        # Concatenate image and text embeddings for a combined vector
        combined_embedding = torch.cat([image_embeddings, text_embeddings], dim=-1)

        # Convert embeddings to list format for JSON response
        image_embedding_list = image_embeddings.cpu().numpy().tolist()
        text_embedding_list = text_embeddings.cpu().numpy().tolist()
        combined_embedding_list = combined_embedding.cpu().numpy().tolist()

        # Return the embeddings and the combined one
        return {
            "image_embedding": image_embedding_list,
            "text_embeddings": text_embedding_list,
            "combined_embedding": combined_embedding_list
        }
