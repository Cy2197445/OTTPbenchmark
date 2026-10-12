'Precursor baseline zoo and calibration utilities.'

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class PrecursorModelSpec:
    'Static metadata for one baseline family.'

    model_name: str
    model_family: str
    input_view: str
    trainable: bool


MODEL_REGISTRY: dict[str, PrecursorModelSpec] = {
    "gru": PrecursorModelSpec(model_name="gru", model_family="sequential", input_view="sequence", trainable=True),
    "temporal-cnn": PrecursorModelSpec(
        model_name="temporal-cnn",
        model_family="sequential",
        input_view="sequence",
        trainable=True,
    ),
    "transformer": PrecursorModelSpec(
        model_name="transformer",
        model_family="sequential",
        input_view="sequence",
        trainable=True,
    ),
    "flat-mlp": PrecursorModelSpec(
        model_name="flat-mlp",
        model_family="nonsequential",
        input_view="flat",
        trainable=True,
    ),
    "linear": PrecursorModelSpec(model_name="linear", model_family="nonsequential", input_view="flat", trainable=True),
    "prevalence": PrecursorModelSpec(
        model_name="prevalence",
        model_family="reference",
        input_view="reference",
        trainable=False,
    ),
}


def get_precursor_model_spec(model_name: str) -> PrecursorModelSpec:
    'Return the registered metadata for a baseline.'
    try:
        return MODEL_REGISTRY[str(model_name)]
    except KeyError as exc:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unsupported model `{model_name}`. Supported models: {supported}.") from exc


def _make_classification_head(hidden_dim: int, num_classes: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(int(hidden_dim)),
        nn.Linear(int(hidden_dim), int(hidden_dim)),
        nn.ReLU(),
        nn.Linear(int(hidden_dim), int(num_classes)),
    )


class GRUPrecursorModel(nn.Module):
    'Causal GRU classifier over fixed-length prefix windows.'

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        gru_dropout = float(dropout) if int(num_layers) > 1 else 0.0
        self.gru = nn.GRU(
            input_size=int(input_dim),
            hidden_size=int(hidden_dim),
            num_layers=int(num_layers),
            batch_first=True,
            dropout=gru_dropout,
        )
        self.head = _make_classification_head(hidden_dim=int(hidden_dim), num_classes=int(num_classes))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(sequence)
        final_hidden = hidden[-1]
        return self.head(final_hidden)


class CausalTransformerPrecursorModel(nn.Module):
    'Strictly causal Transformer classifier over fixed-length prefixes.'

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        num_layers: int = 2,
        num_heads: int = 4,
        ffn_multiplier: int = 4,
        dropout: float = 0.0,
        max_length: int = 32,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        if int(d_model) % int(num_heads) != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}.")
        if int(max_length) <= 0:
            raise ValueError("max_length must be positive.")
        self.input_dim = int(input_dim)
        self.d_model = int(d_model)
        self.max_length = int(max_length)
        self.input_projection = nn.Linear(int(input_dim), int(d_model))
        self.position_embedding = nn.Embedding(int(max_length), int(d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(num_heads),
            dim_feedforward=int(ffn_multiplier) * int(d_model),
            dropout=float(dropout),
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.head = _make_classification_head(hidden_dim=int(d_model), num_classes=int(num_classes))

    def encode(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 3:
            raise ValueError(f"sequence must have shape [batch, time, features], got {tuple(sequence.shape)}")
        if int(sequence.shape[-1]) != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {int(sequence.shape[-1])}.")
        length = int(sequence.shape[1])
        if length <= 0 or length > self.max_length:
            raise ValueError(f"Sequence length must be in [1, {self.max_length}], got {length}.")
        positions = torch.arange(length, device=sequence.device, dtype=torch.long)
        encoded = self.input_projection(sequence) + self.position_embedding(positions).unsqueeze(0)
        causal_mask = torch.triu(
            torch.ones((length, length), dtype=torch.bool, device=sequence.device),
            diagonal=1,
        )
        return self.encoder(encoded, mask=causal_mask)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(sequence)[:, -1, :])


def trainable_parameter_count(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad))


def select_transformer_d_model(
    input_dim: int,
    *,
    gru_hidden_dim: int = 128,
    num_classes: int = 3,
    window_size: int = 32,
    candidates: tuple[int, ...] = tuple(range(16, 129, 4)),
) -> dict[str, int | float]:
    'Performance-blind parameter matching required by OTTP-R26-G1-v1.'
    gru = GRUPrecursorModel(
        input_dim=int(input_dim),
        hidden_dim=int(gru_hidden_dim),
        num_layers=1,
        dropout=0.0,
        num_classes=int(num_classes),
    )
    gru_count = trainable_parameter_count(gru)
    choices: list[tuple[int, int, int]] = []
    for d_model in candidates:
        transformer = CausalTransformerPrecursorModel(
            input_dim=int(input_dim),
            d_model=int(d_model),
            num_layers=2,
            num_heads=4,
            ffn_multiplier=4,
            dropout=0.0,
            max_length=int(window_size),
            num_classes=int(num_classes),
        )
        count = trainable_parameter_count(transformer)
        choices.append((abs(count - gru_count), int(d_model), count))
    _, selected, transformer_count = min(choices, key=lambda item: (item[0], item[1]))
    relative_deviation_percent = 100.0 * (float(transformer_count) - float(gru_count)) / float(gru_count)
    return {
        "input_dim": int(input_dim),
        "gru_parameter_count": int(gru_count),
        "d_model": int(selected),
        "transformer_parameter_count": int(transformer_count),
        "relative_deviation_percent": float(relative_deviation_percent),
    }


class _CausalConvBlock(nn.Module):
    'One causal temporal-convolution block.'

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.conv = nn.Conv1d(int(in_channels), int(out_channels), kernel_size=int(kernel_size))
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        padded = F.pad(inputs, (self.kernel_size - 1, 0))
        outputs = self.conv(padded)
        outputs = self.activation(outputs)
        return self.dropout(outputs)


class TemporalCNNPrecursorModel(nn.Module):
    'Causal temporal-CNN classifier over prefix windows.'

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        num_classes: int = 3,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if int(kernel_size) <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}.")
        blocks: list[nn.Module] = []
        in_channels = int(input_dim)
        for _ in range(int(num_layers)):
            blocks.append(
                _CausalConvBlock(
                    in_channels=in_channels,
                    out_channels=int(hidden_dim),
                    kernel_size=int(kernel_size),
                    dropout=float(dropout),
                )
            )
            in_channels = int(hidden_dim)
        self.encoder = nn.Sequential(*blocks)
        self.head = _make_classification_head(hidden_dim=int(hidden_dim), num_classes=int(num_classes))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(sequence.transpose(1, 2))
        final_hidden = encoded[:, :, -1]
        return self.head(final_hidden)


class FlatMLPPrecursorModel(nn.Module):
    'Flattened-window MLP classifier.'

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        num_classes: int = 3,
        mlp_depth: int = 2,
    ) -> None:
        super().__init__()
        if int(mlp_depth) <= 0:
            raise ValueError(f"mlp_depth must be positive, got {mlp_depth}.")
        layers: list[nn.Module] = [nn.LayerNorm(int(input_dim))]
        current_dim = int(input_dim)
        for _ in range(int(mlp_depth)):
            layers.extend(
                [
                    nn.Linear(int(current_dim), int(hidden_dim)),
                    nn.ReLU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            current_dim = int(hidden_dim)
        layers.append(nn.Linear(int(current_dim), int(num_classes)))
        self.network = nn.Sequential(*layers)

    def forward(self, flat_features: torch.Tensor) -> torch.Tensor:
        return self.network(flat_features)


class LinearPrecursorModel(nn.Module):
    'Single-layer linear classifier over flattened windows.'

    def __init__(self, input_dim: int, num_classes: int = 3) -> None:
        super().__init__()
        self.linear = nn.Linear(int(input_dim), int(num_classes))

    def forward(self, flat_features: torch.Tensor) -> torch.Tensor:
        return self.linear(flat_features)


@dataclass
class PrevalenceBaseline:
    'Reference baseline that predicts train-label prevalence.'

    num_classes: int = 3
    class_probabilities: torch.Tensor | None = None

    def fit(self, labels: torch.Tensor | Any) -> torch.Tensor:
        arr = torch.as_tensor(labels, dtype=torch.long).reshape(-1)
        if arr.numel() <= 0:
            probs = torch.full((int(self.num_classes),), 1.0 / float(self.num_classes), dtype=torch.float32)
        else:
            counts = torch.bincount(arr, minlength=int(self.num_classes)).to(torch.float32)
            total = counts.sum()
            probs = counts / total if float(total.item()) > 0 else torch.full_like(counts, 1.0 / float(self.num_classes))
        self.class_probabilities = probs
        return probs

    def predict_proba(self, batch_size: int, device: torch.device | None = None) -> torch.Tensor:
        if self.class_probabilities is None:
            raise RuntimeError("PrevalenceBaseline must be fitted before calling predict_proba.")
        probs = self.class_probabilities.to(device=device or self.class_probabilities.device)
        return probs.unsqueeze(0).repeat(int(batch_size), 1)


def build_precursor_model(
    model_name: str,
    *,
    input_dim: int,
    hidden_dim: int = 128,
    num_layers: int = 1,
    dropout: float = 0.0,
    num_classes: int = 3,
    cnn_kernel_size: int = 3,
    mlp_depth: int = 2,
    transformer_d_model: int | None = None,
    transformer_num_layers: int = 2,
    transformer_num_heads: int = 4,
    transformer_ffn_multiplier: int = 4,
    transformer_max_length: int = 32,
) -> nn.Module | PrevalenceBaseline:
    'Instantiate one registered baseline.'
    spec = get_precursor_model_spec(model_name)
    if spec.model_name == "gru":
        return GRUPrecursorModel(
            input_dim=int(input_dim),
            hidden_dim=int(hidden_dim),
            num_layers=int(num_layers),
            dropout=float(dropout),
            num_classes=int(num_classes),
        )
    if spec.model_name == "temporal-cnn":
        return TemporalCNNPrecursorModel(
            input_dim=int(input_dim),
            hidden_dim=int(hidden_dim),
            num_layers=int(num_layers),
            dropout=float(dropout),
            num_classes=int(num_classes),
            kernel_size=int(cnn_kernel_size),
        )
    if spec.model_name == "transformer":
        resolved_d_model = (
            int(transformer_d_model)
            if transformer_d_model is not None
            else int(
                select_transformer_d_model(
                    int(input_dim),
                    gru_hidden_dim=int(hidden_dim),
                    num_classes=int(num_classes),
                    window_size=int(transformer_max_length),
                )["d_model"]
            )
        )
        return CausalTransformerPrecursorModel(
            input_dim=int(input_dim),
            d_model=resolved_d_model,
            num_layers=int(transformer_num_layers),
            num_heads=int(transformer_num_heads),
            ffn_multiplier=int(transformer_ffn_multiplier),
            dropout=float(dropout),
            max_length=int(transformer_max_length),
            num_classes=int(num_classes),
        )
    if spec.model_name == "flat-mlp":
        return FlatMLPPrecursorModel(
            input_dim=int(input_dim),
            hidden_dim=int(hidden_dim),
            dropout=float(dropout),
            num_classes=int(num_classes),
            mlp_depth=int(mlp_depth),
        )
    if spec.model_name == "linear":
        return LinearPrecursorModel(input_dim=int(input_dim), num_classes=int(num_classes))
    if spec.model_name == "prevalence":
        return PrevalenceBaseline(num_classes=int(num_classes))
    raise ValueError(f"Unhandled model_name={model_name}")


@dataclass
class TemperatureScaler:
    'Simple scalar temperature calibration for logits.'

    temperature: float = 1.0

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50) -> float:
        'Fit temperature on validation logits.'
        if logits.ndim != 2:
            raise ValueError("logits must be a rank-2 tensor.")
        if labels.ndim != 1:
            raise ValueError("labels must be a rank-1 tensor.")
        if logits.shape[0] != labels.shape[0]:
            raise ValueError("logits and labels must have the same batch size.")
        if logits.shape[0] <= 0:
            self.temperature = 1.0
            return self.temperature

        log_t = torch.zeros(1, device=logits.device, requires_grad=True)
        optimizer = torch.optim.LBFGS([log_t], max_iter=int(max_iter), line_search_fn="strong_wolfe")
        criterion = nn.CrossEntropyLoss()

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            temperature = torch.exp(log_t).clamp(min=1e-4)
            loss = criterion(logits / temperature, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = float(torch.exp(log_t).clamp(min=1e-4).item())
        return self.temperature

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        'Scale logits by the fitted temperature.'
        temperature = max(float(self.temperature), 1e-4)
        return logits / temperature
