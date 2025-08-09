from dataclasses import dataclass
import torch, tqdm, os
from typing import Any
from ..core.base_model import BaseModelWrapper
from ..core.base_dataset import BaseDatasetAdapter

@dataclass
class TrainConfig:
    epochs: int = 3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True
    ckpt_dir: str = "artifacts/checkpoints"
    ckpt_name: str = "model.pt"

class Trainer:
    def __init__(self, model_wrapper: BaseModelWrapper, dataset_adapter: BaseDatasetAdapter, cfg: TrainConfig):
        self.mw = model_wrapper
        self.ds = dataset_adapter
        self.cfg = cfg
        self.net = self.mw.build().to(cfg.device)
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    def fit(self) -> None:
        self.net.train()
        for ep in range(self.cfg.epochs):
            pbar = tqdm.tqdm(self.ds.train_loader(), desc=f"epoch {ep+1}/{self.cfg.epochs}")
            for batch in pbar:
                batch = {k: (v.to(self.cfg.device) if hasattr(v, "to") else v) for k,v in batch.items()}
                with torch.cuda.amp.autocast(enabled=self.cfg.amp):
                    pred = self.mw.forward(batch)
                    loss = self.mw.compute_loss(pred, batch)
                self.opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                self.scaler.update()
                pbar.set_postfix(loss=float(loss))

        os.makedirs(self.cfg.ckpt_dir, exist_ok=True)
        path = os.path.join(self.cfg.ckpt_dir, self.cfg.ckpt_name)
        self.mw.save(path)
        print(f"Saved checkpoint to {path}")
