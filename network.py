import torch
from torch.nn import Linear
from transformers import AutoModel

class TextClassificationModel(torch.nn.Module):
    """Defines a Pytorch text classification bert based model."""

    def __init__(self, num_labels: int):
        super().__init__()
        self.feature_extractor = AutoModel.from_pretrained("distilbert-base-uncased")
        self.classifier = Linear(self.feature_extractor.config.hidden_size, num_labels)
        print("Hidden size: ", self.feature_extractor.config.hidden_size)

    def forward(self, x, attention_mask):
        """Model forward function."""
        encoded_layers = self.feature_extractor(
            input_ids=x, attention_mask=attention_mask
        ).last_hidden_state
        classification_embedding = encoded_layers[:, 0]
        logits = self.classifier(classification_embedding)

        return logits
