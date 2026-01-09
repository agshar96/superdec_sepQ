# test.py
import os
import torch
import torch.distributed as dist
import hydra
from omegaconf import DictConfig
from tester import Tester
from utils import (
    build_model,
    build_test_dataloader,
    build_loss,
    set_seed,
    setup_ddp,
    is_main_process,
)

try:
    import wandb
except ImportError:
    wandb = None


@hydra.main(config_path="../configs", config_name="eval", version_base=None)
def main(cfg: DictConfig):
    is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if is_distributed:
        print(f"You don't need distributed for testing. It is recommended to run on a single GPU.")

    if is_distributed:
        local_rank = setup_ddp()
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = 0
        device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    set_seed(cfg.seed + local_rank)

    run = None
    if cfg.use_wandb and wandb is not None and is_main_process():
        run = wandb.init(
            project=cfg.wandb.project,
            name=cfg.run_name,
        )

    # Build model
    model = build_model(cfg).to(device)

    # Load checkpoint
    ckp_path = os.path.join(cfg.checkpoints_folder, cfg.checkpoint_file)
    checkpoint = torch.load(ckp_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=False
        )

    # Build dataloaders (reuse val as test or add explicit test split in utils)
    dataloaders, _ = build_test_dataloader(cfg, is_distributed=False)
    test_loader = dataloaders['test']

    loss_fn = build_loss(cfg).to(device)

    tester = Tester(
        model=model,
        dataloader=test_loader,
        loss_fn=loss_fn,
        wandb_run=run,
        is_distributed=is_distributed,
    )

    metrics = tester.test()

    if is_main_process():
        print("\n===== Test Results =====")
        for k, v in metrics.items():
            print(f"{k}: {v:.6f}")

    if run is not None:
        run.finish()

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
