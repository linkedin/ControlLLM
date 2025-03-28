from __future__ import annotations
from typing import Any, List
import torch
from torch import Tensor, nn
from sentence_transformers import util
from sentence_transformers.SentenceTransformer import SentenceTransformer


class MarginCoSENTLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        scale: float = 4.0,  # Adjusted default scale based on calibration analysis.
        margin_scale: float = 1.0,  # Factor to weight the label difference (margin).
        similarity_fct=util.pairwise_cos_sim,
    ) -> None:
        """
        Margin-enhanced CoSENT Loss that leverages the magnitude of label differences.

        Expect pairs of sentence embeddings(prompt/query, repsonse/document) and their corresponding labels.
        For a pair i, let s_i = scale * cosine_similarity.
        For labels y_i (e.g., in {0,1,2,3,4}), the margin between any two pairs is (y_i - y_j).
        For data preprocessing, refer to example of ./data/semantic_search_cosent_dataset.py.

        The enhanced loss for valid pairs (where y_i > y_j) is defined as:
            L = log( 1 + sum_{i,j: y_i > y_j} exp(- ((s_i - s_j) - margin_scale*(y_i - y_j)))

        Args:
            model: SentenceTransformer model.
            scale: Scale factor for cosine differences.
            margin_scale: Scale factor for the ground truth margin differences. Set to 0 to disable margin.
            similarity_fct: Function to compute pairwise cosine similarity.
        """
        super().__init__()
        self.model = model
        self.similarity_fct = similarity_fct
        self.scale = scale
        self.margin_scale = margin_scale

    def forward(self, sentence_features: List[dict[str, Tensor]], label: Tensor, return_verbose: bool = False) -> Tensor:
        # Compute embeddings for each input using the model.
        embeddings = [feat["embeddings"] if "embeddings" in feat else self.model(feat)["sentence_embedding"] for feat in sentence_features]

        # Compute cosine similarity scores and scale them.
        similarity_prompt_response = self.similarity_fct(embeddings[0], embeddings[1]) * self.scale

        # Compute pairwise differences of predicted scores: Δs_{ij} = s_i - s_j.
        score_diffs = similarity_prompt_response[:, None] - similarity_prompt_response[None, :]

        # Compute pairwise differences of labels (the ground-truth margin): m_{ij} = margin_scale * (y_i - y_j).
        label_diffs = (label[:, None] - label[None, :]) * self.margin_scale

        # Create mask for valid pairs: only consider pairs where y_i > y_j.
        mask = (label[:, None] > label[None, :]).float()

        # Subtract the desired margin from the predicted difference.
        margin_enhanced_diffs = label_diffs - score_diffs

        # For invalid pairs, subtract a large constant so they contribute negligibly.
        margin_enhanced_diffs = margin_enhanced_diffs - (1 - mask) * 1e12

        # Append a zero for numerical stability (so that exp(0)=1 is always included).
        margin_enhanced_diffs = torch.cat(
            (torch.zeros(1, device=similarity_prompt_response.device), margin_enhanced_diffs.view(-1)), dim=0
        )

        loss = torch.logsumexp(margin_enhanced_diffs, dim=0)

        if return_verbose:
            return (loss, {"similarity_prompt_response": similarity_prompt_response / self.scale})
        else:
            return loss

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "margin_scale": self.margin_scale,
            "similarity_fct": self.similarity_fct.__name__,
        }

    @property
    def citation(self) -> str:
        return """
@online{kexuefm-margin-enhanced-cosent,
    title={Margin-Enhanced CoSENT Loss for Sentence Embeddings},
    author={Your Name},
    year={2025},
    url={https://your-citation-url.example.com},
}
"""


# Example usage (dummy data):
if __name__ == "__main__":
    class DummyModel:
        def __call__(self, features: dict[str, Tensor]) -> dict[str, Tensor]:
            batch_size = features["input_ids"].shape[0]
            return {"sentence_embedding": torch.randn(batch_size, 768)}

    dummy_model = DummyModel()
    loss_fn = MarginCoSENTLoss(model=dummy_model, scale=4.0, margin_scale=1.0)

    batch_size = 4
    seq_length = 10
    dummy_features_A = {"input_ids": torch.randint(0, 1000, (batch_size, seq_length))}
    dummy_features_B = {"input_ids": torch.randint(0, 1000, (batch_size, seq_length))}
    dummy_labels = torch.tensor([3.0, 1.0, 2.0, 4.0])

    loss = loss_fn([dummy_features_A, dummy_features_B], dummy_labels)
    print("Loss:", loss.item())
