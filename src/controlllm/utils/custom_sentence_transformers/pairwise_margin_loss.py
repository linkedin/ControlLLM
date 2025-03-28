from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch import Tensor, nn

from sentence_transformers import util
from sentence_transformers.SentenceTransformer import SentenceTransformer


# Designed as a torch module following the SentenceTransformer's loss function template.
class MarginPairwiseLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        scale: float = 4.0,
        margin_scale: float = 1.0,  # Factor to weight the label difference (margin).
        similarity_fct=util.pairwise_cos_sim,
    ) -> None:
        """
        This loss function implements a margin-based logistic loss for training sentence embeddings.

        Each training example is provided as a single dictionary containing:
            - 'prompt_input_ids' and 'prompt_attention_mask' for the anchor,
            - 'chosen_input_ids' and 'chosen_attention_mask' for the positive example,
            - 'rejected_input_ids' and 'rejected_attention_mask' for the negative example,
            - 'label' is the margin value, computed as:
                  margin = (score_chosen: ground truth similarity of (prompt, rejected)) - (score_reject: ground truth similarity of (prompt, chosen))
            - 'score_chosen': the ground truth similarity score between the prompt and the chosen example.
            - 'score_rejected': the ground truth similarity score between the prompt and the rejected example.
        For data preprocessing, refer to example of ./data/semantic_search_dataset.py.

        Why scale: float = 4.0:
            Given that you want the model to learn to calibrate the cosine scores so that they directly reflect the ground-truth differences (with original scores in \([0,1,2,3,4]\) and thus margins in \([1,2,3,4]\)), we need the predicted differences to match these margins.
            For example, if we linearly map the original scores to the \([0,1]\) range—i.e.
            \[
            \text{cosine}_{\text{ideal}} = \frac{\text{score}}{4}
            \]
            then for a prompt–chosen pair with ground-truth score 3 and a prompt–rejected pair with score 1 we get:
            \[
            s_{QD^pos} = \frac{3}{4} = 0.75, \quad s_{QD^neg} = \frac{1}{4} = 0.25.
            \]
            Their difference is:
            \[
            s_{QD^pos} - s_{QD^neg} = 0.75 - 0.25 = 0.5.
            \]
            However, the ground-truth margin is:
            \[
            3 - 1 = 2.
            \]
            To have the loss (e.g.
            \[
            L = \log\left(1 + \exp(s_{QD^pos} - s_{QD^neg} - \text{margin})\right)
            \]
            ) be near zero when the model is perfect, we need the ideal difference \( s_{QD^pos} - s_{QD^neg} \) to equal the margin (i.e., 2 in this case).
            Since our unscaled difference is only 0.5, we need to “stretch” the difference by a scale factor \( \alpha \) such that:
            \[
            \alpha \times 0.5 = 2 \Rightarrow \alpha = 4.
            \]
            Thus, using a default scale of 4 is a good choice—it will map the ideal unscaled cosine differences (assuming outputs in \([0,1]\)) to the margin range. In other words, with scale 4, if the model perfectly predicts the calibrated values, then:
            \[
            \text{scaled } s_{QD^pos} - \text{scaled } s_{QD^neg} = 4 \times (s_{QD^pos} - s_{QD^neg}) = \text{margin}.
            \]
            This way the loss
            \[
            L = \log\left(1 + \exp(- (4(s_{QD^pos} - s_{QD^neg}) - \text{margin})\right))
            \]
            will be near zero when the predicted differences match the ground truth margins.

            The model computes cosine similarities:
                s_QD^pos = cos(embedding_prompt, embedding_chosen)
                s_QD^neg = cos(embedding_prompt, embedding_rejected)
            
            The loss for each triple is defined as:
                L = log(1 + exp(s_QD^pos - s_QD^neg - margin))
            which encourages the model to have s_QD^neg exceed s_QD^pos by at least the margin.

        Args:
            model: SentenceTransformer model to compute embeddings.
            similarity_fct: Function to compute cosine similarity between embeddings.
                Default is `util.pairwise_cos_sim`.
            scale: A scaling factor (acting as inverse temperature) applied to the cosine similarities.
                Default is 4.0 as margin value is [1, 2, 3, 4] and util.pairwise_cos_sim is in [-1, 1] but assuming it is in [0, 1] range due to our training data is mostly positive and hard negtive.
            margin_scale: A scaling factor applied to the ground truth margin values. . Set to 0 to disable margin.
            similarity_fct: Function to compute pairwise cosine similarity.
        """
        super().__init__()
        self.model = model
        self.similarity_fct = similarity_fct
        self.scale = scale
        self.margin_scale = margin_scale

    def forward(self, sentence_features: List[dict[str, Tensor]], label: Tensor, return_verbose: bool = False) -> Tensor:
        """
        Args:
            sentence_features: List of dictionary containing:
                prompt_features:
                - 'input_ids': Tensor of shape (batch_size, seq_length)
                - 'attention_mask': Tensor of shape (batch_size, seq_length)
                chosen_features:
                - 'input_ids': Tensor of shape (batch_size, seq_length)
                - 'attention_mask': Tensor of shape (batch_size, seq_length)
                - 'embeddings': Tensor of shape (batch_size, embedding_dim)  # Optional
                rejected_features:
                - 'rejected_input_ids': Tensor of shape (batch_size, seq_length)
                - 'rejected_attention_mask': Tensor of shape (batch_size, seq_length)
                - 'embeddings': Tensor of shape (batch_size, embedding_dim)  # Optional
            label:
                - 'label': Tensor of shape (batch_size,) representing the margin values
                           (i.e. ground truth similarity difference: s(prompt,chosen) - s(prompt,rejected))

            return_verbose: If True, return a tuple with the loss and a dictionary of verbose outputs. This is for evaluation purposes.

        Returns:
            A scalar tensor representing the mean loss over the batch.
        """
        # Compute embeddings for each input using the model.
        embeddings = [feat["embeddings"] if "embeddings" in feat else self.model(feat)["sentence_embedding"] for feat in sentence_features]

        # Run a forward pass through the model for each set of features
        emb_prompt = embeddings[0]
        emb_chosen = embeddings[1]
        emb_rejected = embeddings[2]

        # Compute cosine similarities (optionally scaled)
        similarity_prompt_chosen = self.similarity_fct(emb_prompt, emb_chosen) * self.scale
        similarity_prompt_rejected = self.similarity_fct(emb_prompt, emb_rejected) * self.scale

        # Compute the logistic loss for each triple:
        # L = log(1 + exp(similarity_prompt_chosen - similarity_prompt_rejected - margin))
        losses = torch.log1p(torch.exp(self.margin_scale * label - (similarity_prompt_chosen - similarity_prompt_rejected)))
        loss = losses.mean()

        if return_verbose:
            return (loss, {"similarity_prompt_chosen": similarity_prompt_chosen / self.scale, "similarity_prompt_rejected": similarity_prompt_rejected / self.scale})
        else:
            return loss

    def get_config_dict(self) -> Dict[str, Any]:
        return {"scale": self.scale, "similarity_fct": self.similarity_fct.__name__}

    @property
    def citation(self) -> str:
        return """
            @online{hc-XXXX,
                title={Margin-based Cosine Loss for Sentence Embeddings},
                author={Name},
                year={2025},
                url={https://your-citation-url.example.com},
            }
            """


# Example usage:
if __name__ == "__main__":
    # A dummy model for demonstration purposes
    class DummyModel:
        def __call__(self, features):
            batch_size = features["input_ids"].shape[0]
            return {"sentence_embedding": torch.randn(batch_size, 768)}

    dummy_model = DummyModel()
    loss_fn = MarginPairwiseLoss(model=dummy_model)

    # Create dummy data for a batch (here batch_size=2 for illustration)
    batch_size = 2
    seq_length = 10  # dummy sequence length
    dummy_features = {
        "prompt_input_ids": torch.randint(0, 1000, (batch_size, seq_length)),
        "prompt_attention_mask": torch.ones(batch_size, seq_length),
        "chosen_input_ids": torch.randint(0, 1000, (batch_size, seq_length)),
        "chosen_attention_mask": torch.ones(batch_size, seq_length),
        "rejected_input_ids": torch.randint(0, 1000, (batch_size, seq_length)),
        "rejected_attention_mask": torch.ones(batch_size, seq_length),
        "label": torch.randn(batch_size)  # dummy margin values
    }

    loss = loss_fn(dummy_features)
    print("Loss:", loss.item())
