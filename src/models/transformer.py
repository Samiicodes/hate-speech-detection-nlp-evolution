from __future__ import annotations

import math
import os
import random
import tempfile
import time
from pathlib import Path

import numpy as np

from ..config import CACHE_DIR, DEFAULT_TASK, SEED, TaskConfig
from ..harness import BaseDetector

# torch optional: systems 1-3 skip it
try:  # pragma: no cover - exercised only when the deps are installed
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    Dataset = object  # type: ignore[assignment,misc]


def require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "system 4 needs torch and transformers:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cu128\n"
            "  pip install transformers accelerate"
        )


def set_all_seeds(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def _stratified_subsample(texts, labels, weights, fraction: float, seed: int):
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(texts))
    keep, _ = train_test_split(
        idx, train_size=fraction, random_state=seed, stratify=np.asarray(labels)
    )
    keep = np.sort(keep)
    sub_texts = [texts[i] for i in keep]
    sub_labels = [labels[i] for i in keep]
    sub_weights = None if weights is None else [weights[i] for i in keep]
    return sub_texts, sub_labels, sub_weights


class _TextDataset(Dataset):

    def __init__(
        self,
        encodings: dict,
        labels: list[int] | None = None,
        weights: list[float] | None = None,
    ) -> None:
        self.encodings = encodings
        self.labels = labels
        self.weights = weights

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> dict:
        item = {k: v[idx] for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.weights is not None:
            item["weight"] = torch.tensor(self.weights[idx], dtype=torch.float32)
        return item


class TransformerDetector(BaseDetector):

    def __init__(
        self,
        name: str = "roberta_base",
        model_name: str = "roberta-base",
        seed: int = SEED,
        epochs: int = 3,
        lr: float = 2e-5,
        batch_size: int = 16,
        eval_batch_size: int = 64,
        max_length: int = 128,
        warmup_ratio: float = 0.06,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        class_weighted: bool = True,
        task: TaskConfig = DEFAULT_TASK,
        val_data: tuple[list[str], list[int]] | None = None,
        select_best_epoch: bool = True,
        freeze_encoder: bool = False,
        train_fraction: float = 1.0,
        device: str | None = None,
        use_amp: bool = True,
        deterministic: bool = False,
        cache_dir: Path | None = None,
    ) -> None:
        self.name = name
        self.model_name = model_name
        self.seed = seed
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.max_length = max_length
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.class_weighted = class_weighted
        self.task = task
        self.n_labels = task.n_classes
        self.val_data = val_data
        self.select_best_epoch = select_best_epoch
        # Dose-response controls
        self.freeze_encoder = freeze_encoder
        self.train_fraction = train_fraction
        self.n_train_used: int | None = None
        self.n_trainable: int | None = None
        self.use_amp = use_amp
        self.deterministic = deterministic
        self.cache_dir = cache_dir or (CACHE_DIR / "hf")

        self._device_arg = device
        self.device = None
        self.model = None
        self.tokenizer = None
        self.epoch_history: list[dict] = []
        self.best_epoch: int | None = None
        self._n_params: int | None = None

    # setup

    def _resolve_device(self) -> str:
        if self._device_arg:
            return self._device_arg
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _encode(self, texts: list[str]) -> dict:
        enc = self.tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return dict(enc)

    # training

    # The harness may pass per-instance weights; see src/reweight.py.
    supports_sample_weight = True

    def fit(
        self,
        texts: list[str],
        labels: list[int],
        sample_weight: list[float] | None = None,
    ) -> None:
        require_torch()
        set_all_seeds(self.seed, self.deterministic)
        self.device = self._resolve_device()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, cache_dir=str(self.cache_dir)
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.n_labels, cache_dir=str(self.cache_dir)
        ).to(self.device)
        self._n_params = sum(p.numel() for p in self.model.parameters())

        # Freeze all but the head
        if self.freeze_encoder:
            for pname, param in self.model.named_parameters():
                param.requires_grad = "classifier" in pname
        self.n_trainable = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        # Seeded stratified subsample
        if self.train_fraction < 1.0:
            texts, labels, sample_weight = _stratified_subsample(
                texts, labels, sample_weight, self.train_fraction, self.seed
            )
        self.n_train_used = len(texts)

        self.weighted = sample_weight is not None
        train_ds = _TextDataset(
            self._encode(texts), list(labels),
            None if sample_weight is None else [float(w) for w in sample_weight],
        )
        # seeded shuffle
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True, generator=generator,
        )

        # inverse-frequency class weights, matching system 3
        loss_weights = None
        if self.class_weighted:
            counts = np.bincount(
                np.asarray(labels, dtype=int), minlength=self.n_labels
            )
            weights = len(labels) / (self.n_labels * np.maximum(counts, 1))
            loss_weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        # none, so weights apply per example
        loss_fn = torch.nn.CrossEntropyLoss(weight=loss_weights, reduction="none")

        # no weight decay on bias or LayerNorm
        decay, no_decay = [], []
        for pname, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            (no_decay if pname.endswith("bias") or "LayerNorm" in pname else decay
             ).append(param)
        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.lr,
        )

        total_steps = max(1, len(loader) * self.epochs)
        warmup_steps = int(total_steps * self.warmup_ratio)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            remaining = total_steps - warmup_steps
            return max(0.0, (total_steps - step) / max(1, remaining))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        amp_on = self.use_amp and self.device == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp_on)

        best_score, best_state = -1.0, None
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            running_loss, n_batches = 0.0, 0
            epoch_start = time.perf_counter()

            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                targets = batch.pop("labels")
                weights = batch.pop("weight", None)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=amp_on):
                    logits = self.model(**batch).logits
                    per_example = loss_fn(logits, targets)
                    loss = (
                        per_example.mean() if weights is None
                        else (per_example * weights).mean()
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                running_loss += float(loss.detach())
                n_batches += 1

            record = {
                "epoch": epoch,
                "train_loss": running_loss / max(1, n_batches),
                "seconds": time.perf_counter() - epoch_start,
            }

            if self.val_data is not None:
                from sklearn.metrics import f1_score

                val_texts, val_labels = self.val_data
                preds = self._predict_raw(val_texts)
                score = float(
                    f1_score(val_labels, preds, average="macro", zero_division=0)
                )
                record["val_macro_f1"] = score
                if self.select_best_epoch and score > best_score:
                    best_score, self.best_epoch = score, epoch
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in self.model.state_dict().items()
                    }

            self.epoch_history.append(record)

        # restore the best-scoring epoch rather than the last one
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

    # inference

    def _predict_raw(self, texts: list[str]) -> list[int]:
        self.model.eval()
        ds = _TextDataset(self._encode(texts))
        loader = DataLoader(ds, batch_size=self.eval_batch_size, shuffle=False)
        amp_on = self.use_amp and self.device == "cuda"

        out: list[int] = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with torch.amp.autocast("cuda", enabled=amp_on):
                    logits = self.model(**batch).logits
                out.extend(logits.argmax(dim=-1).cpu().tolist())
        return out

    def predict(self, texts: list[str]) -> list[int]:
        if self.model is None:
            raise RuntimeError(f"{self.name}: fit() must be called before predict()")
        return [int(v) for v in self._predict_raw(list(texts))]

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError(f"{self.name}: fit() must be called first")
        self.model.eval()
        ds = _TextDataset(self._encode(list(texts)))
        loader = DataLoader(ds, batch_size=self.eval_batch_size, shuffle=False)
        probs: list[list[float]] = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                logits = self.model(**batch).logits.float()
                probs.extend(torch.softmax(logits, dim=-1).cpu().tolist())
        return np.asarray(probs)

    # harness hooks

    def params(self) -> dict:
        return {
            "model_name": self.model_name,
            "task": self.task.name,
            "num_labels": self.n_labels,
            "seed": self.seed,
            "epochs": self.epochs,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "class_weighted": self.class_weighted,
            "deterministic": self.deterministic,
            "device": self.device,
            "mixed_precision": self.use_amp and self.device == "cuda",
            "n_parameters": self._n_params,
            "n_trainable_parameters": self.n_trainable,
            "freeze_encoder": self.freeze_encoder,
            "train_fraction": self.train_fraction,
            "n_train_examples_used": self.n_train_used,
            "best_epoch": self.best_epoch,
            "epoch_history": self.epoch_history,
        }

    def cost_metadata(self) -> dict:
        size = None
        if self.model is not None:
            # fp32 checkpoint size, measured rather than estimated
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "model.pt"
                torch.save(self.model.state_dict(), path)
                size = path.stat().st_size
        return {
            "model_size_bytes": size,
            "api_cost_usd": 0.0,
            "size_note": "fp32 state_dict on disk",
            "gpu": (
                torch.cuda.get_device_name(0)
                if _TORCH_AVAILABLE and torch.cuda.is_available()
                else None
            ),
        }

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)
        return directory


__all__ = ["TransformerDetector", "require_torch", "set_all_seeds"]
