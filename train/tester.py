# tester.py
import torch
from tqdm import tqdm
from utils import is_main_process

try:
    import wandb
except ImportError:
    wandb = None


class Tester:
    def __init__(self, model, dataloader, loss_fn, wandb_run=None, is_distributed=False):
        self.model = model
        self.dataloader = dataloader
        self.loss_fn = loss_fn
        self.wandb_run = wandb_run
        self.is_distributed = is_distributed

    @torch.no_grad()
    def test(self):
        """Run evaluation on the test set."""
        if self.is_distributed and not is_main_process():
            return {}

        self.model.eval()
        pbar = tqdm(self.dataloader, desc="Testing", leave=False)

        total_batches = 0
        avg_loss_dict = {}

        for batch in pbar:
            pc = batch["points"].cuda().float()
            normals = batch["normals"].cuda().float()

            outdict = self.model(pc)
            _, loss_dict = self.loss_fn(pc, normals, outdict)

            total_batches += 1
            for k, v in loss_dict.items():
                avg_loss_dict[k] = avg_loss_dict.get(k, 0.0) + v

            pbar.set_postfix({k: f"{v:.4f}" for k, v in loss_dict.items()})

        for k in avg_loss_dict:
            avg_loss_dict[k] /= total_batches

        if self.wandb_run is not None:
            self.wandb_run.log({f"test/{k}": v for k, v in avg_loss_dict.items()})

        return avg_loss_dict
