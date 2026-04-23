import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import os
import shutil
import sys
import subprocess
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import argparse  # [新增]

# 引入项目模块
from config import Config
from dataloader_ddp import get_dataloader
from model_lstm_cnn import DualEncoderModel
from loss import ContrastiveLoss
from dist_utils import reduce_mean, get_rank, is_dist_avail_and_initialized

nccl_env = {
    "NCCL_TIMEOUT": str(2*3600000), # 2h
    "NCCL_IB_DISABLE": "1",
    "NCCL_P2P_DISABLE": "0",
    "CUDA_LAUNCH_BLOCKING": "0",
}

# -----------------------------------------------------------------------------
# DDP Setup & Utils (保持不变)
# -----------------------------------------------------------------------------

def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        try:
            dist.init_process_group(backend="nccl")
            rank = int(os.environ["RANK"])
            local_rank = int(os.environ["LOCAL_RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            torch.cuda.set_device(local_rank)
            print(f"[Init] Rank {rank}/{world_size} (Local {local_rank}) initialized.")
            return local_rank
        except Exception as e:
            print(f"DDP Init Failed: {e}")
            sys.exit(1)
    else:
        print("Not using Distributed Mode.")
        return 0


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


class AverageMeter:
    def __init__(self): self.reset()

    def reset(self): self.val = 0; self.avg = 0; self.sum = 0; self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def make_next_record_dir(base_dir, prefix='train_'):
    if get_rank() != 0:
        return None
    if not os.path.exists(base_dir): os.makedirs(base_dir)
    existing_dirs = [d for d in os.listdir(base_dir) if d.startswith(prefix)]
    max_idx = -1
    for d in existing_dirs:
        try:
            idx = int(d.split(prefix)[-1].split('_')[0])
            if idx > max_idx: max_idx = idx
        except:
            continue
    new_dir_name = f"{prefix}{max_idx + 1}"
    full_path = os.path.join(base_dir, new_dir_name)
    os.makedirs(full_path, exist_ok=True)
    return full_path


def _run_plot_metrics(ckpt_dir, config):
    if get_rank() != 0: return
    ckpt_path = os.path.join(ckpt_dir, "best_cossim_model.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(ckpt_dir, "best_model.pth")
    if not os.path.exists(ckpt_path): return

    out_dir = os.path.join(ckpt_dir, "plots_vis")
    os.makedirs(out_dir, exist_ok=True)

    plot_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot.py")
    if not os.path.exists(plot_script):
        print("[WARN] plot.py not found.")
        return

    cmd = [
        sys.executable, plot_script,
        "--ckpt", ckpt_path,
        "--out", out_dir,
        "--n", "2000",
        "--umap", "--tsne", "--heatmaps",
    ]
    print(f"[metrics] Running evaluation script...")
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"[metrics] Failed to run plot.py: {e}")


def train_epoch(model, dataloader, optimizer, loss_fn, device, grad_clip, epoch, scaler):
    model.train()
    if hasattr(dataloader.sampler, 'set_epoch'):
        dataloader.sampler.set_epoch(epoch)

    losses = AverageMeter()
    sims = AverageMeter()

    if get_rank() == 0:
        pbar = tqdm(dataloader, desc=f"Train Ep {epoch + 1}")
    else:
        pbar = dataloader

    for inchi_ids, smiles_ids in pbar:
        inchi_ids = inchi_ids.to(device, non_blocking=True)
        smiles_ids = smiles_ids.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast():
            z_inchi, z_smiles = model(inchi_ids, smiles_ids)
            loss = loss_fn(z_inchi, z_smiles)

        if torch.isnan(loss):
            if get_rank() == 0: print("Warning: NaN loss ignored.")
            continue

        scaler.scale(loss).backward()

        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            reduced_loss = reduce_mean(loss)
            z_i_norm = F.normalize(z_inchi.float(), dim=1)
            z_s_norm = F.normalize(z_smiles.float(), dim=1)
            local_sim = (z_i_norm * z_s_norm).sum(dim=1).mean()
            reduced_sim = reduce_mean(local_sim)

        losses.update(reduced_loss.item(), inchi_ids.size(0))
        sims.update(reduced_sim.item(), inchi_ids.size(0))

        if get_rank() == 0:
            pbar.set_postfix({'L': f"{losses.avg:.4f}", 'Sim': f"{sims.avg:.3f}"})

    return losses.avg, sims.avg


def validate(model, dataloader, loss_fn, device):
    model.eval()
    losses = AverageMeter()
    sims = AverageMeter()

    if get_rank() == 0:
        pbar = tqdm(dataloader, desc="Val")
    else:
        pbar = dataloader

    with torch.no_grad():
        for inchi_ids, smiles_ids in pbar:
            inchi_ids = inchi_ids.to(device, non_blocking=True)
            smiles_ids = smiles_ids.to(device, non_blocking=True)

            with autocast():
                z_inchi, z_smiles = model(inchi_ids, smiles_ids)
                loss = loss_fn(z_inchi, z_smiles)

            reduced_loss = reduce_mean(loss)
            z_i_norm = F.normalize(z_inchi.float(), dim=1)
            z_s_norm = F.normalize(z_smiles.float(), dim=1)
            local_sim = (z_i_norm * z_s_norm).sum(dim=1).mean()
            reduced_sim = reduce_mean(local_sim)

            losses.update(reduced_loss.item(), inchi_ids.size(0))
            sims.update(reduced_sim.item(), inchi_ids.size(0))

            if get_rank() == 0:
                pbar.set_postfix({'L': f"{losses.avg:.4f}", 'Sim': f"{sims.avg:.3f}"})

    return losses.avg, sims.avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--local_rank', type=int, default=os.getenv('LOCAL_RANK', -1))
    args = parser.parse_args()

    local_rank = setup_ddp()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    for k, v in nccl_env.items():
        os.environ[k] = v

    conf = Config()

    if get_rank() == 0:
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        print(f"[-] DDP World Size: {world_size}")
        print(f"[-] Per-GPU Batch Size: {conf.batch_size}")

    # 目录设置
    experiment_dir = None
    log_file_path = None

    if get_rank() == 0:
        experiment_dir = make_next_record_dir(conf.save_dir)
        print(f"[-] Save Dir: {experiment_dir}")
        for f in os.listdir('.'):
            if f.endswith('.py'): shutil.copy(f, experiment_dir)
        log_file_path = os.path.join(experiment_dir, 'trainlog.txt')

        if args.resume:
            with open(log_file_path, 'a') as f:
                f.write(f"Resumed from: {args.resume}\n")

    if get_rank() == 0: print("[-] Loading data...")
    train_loader, inchi_tok, smiles_tok = get_dataloader(conf, mode='train')
    val_loader, _, _ = get_dataloader(conf, mode='val')

    model = DualEncoderModel(
        inchi_vocab_size=inchi_tok.get_vocab_size(),
        smiles_vocab_size=smiles_tok.get_vocab_size(),
        config=conf
    ).to(device)
    print(model)

    start_epoch = 0
    best_val_loss = float('inf')
    best_val_sim = -1.0
    best_train_sim = -1.0

    resume_optim_state = None

    if args.resume:
        if os.path.isfile(args.resume):
            if get_rank() == 0:
                print(f"[-] Loading checkpoint from {args.resume} for LARGER dataset training")

            checkpoint = torch.load(args.resume, map_location=device)

            state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict)

            start_epoch = 0

            resume_optim_state = None
            if get_rank() == 0:
                print("    [!] Ignoring previous optimizer state for new dataset adaptation.")

    if is_dist_avail_and_initialized():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        loss_fn = ContrastiveLoss(temperature=conf.temperature).to(device)
    else:
        loss_fn = ContrastiveLoss(temperature=conf.temperature).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
    scaler = GradScaler()
    warmup_epochs = 2

    def incremental_lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        else:

            return 0.98 ** (epoch - warmup_epochs)

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=incremental_lr_lambda,
        last_epoch=-1
    )

    patience = 10
    counter = 0

    if get_rank() == 0:
        print(f"[-] Model Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        print(f"[-] Start training from epoch {start_epoch + 1}...")

    try:
        for epoch in range(start_epoch, conf.epochs):
            if get_rank() == 0:
                # 获取当前 LR 打印
                current_lr = optimizer.param_groups[0]['lr']
                print(f"\nEpoch {epoch + 1}/{conf.epochs} | LR: {current_lr:.8f}")

            t_loss, t_sim = train_epoch(model, train_loader, optimizer, loss_fn, device, conf.grad_clip, epoch, scaler)

            dist.barrier()
            v_loss, v_sim = validate(model, val_loader, loss_fn, device)

            # RoPE Update
            if isinstance(model, DDP):
                if hasattr(model.module, 'step_rope_scheduler'):
                    model.module.step_rope_scheduler(epoch, conf.epochs)
            else:
                if hasattr(model, 'step_rope_scheduler'):
                    model.step_rope_scheduler(epoch, conf.epochs)

            scheduler.step()

            # 保存与打印逻辑
            stop_signal = torch.zeros(1).to(device)

            if get_rank() == 0:
                msg = f"Train: L={t_loss:.4f}, Sim={t_sim:.3f} | Val: L={v_loss:.4f}, Sim={v_sim:.3f}"
                print(msg)
                if log_file_path:
                    with open(log_file_path, 'a') as f:
                        f.write(f"Epoch {epoch + 1}: {msg}\n")

                improved_this_epoch = False
                model_to_save = model.module if isinstance(model, DDP) else model

                # 保存 Best Loss
                if v_loss < best_val_loss:
                    best_val_loss = v_loss
                    torch.save(model_to_save.state_dict(), os.path.join(experiment_dir, 'best_model.pth'))
                    print("  [+] Saved best loss model.")
                    improved_this_epoch = True # 通常我们在意 Sim，所以 Loss 不重置 patience

                if v_sim >= best_val_sim:
                    best_val_sim = v_sim
                    best_train_sim = t_sim
                    torch.save({
                        'epoch': epoch,
                        'model': model_to_save.state_dict(),
                        'optim': optimizer.state_dict(),
                        'sim': v_sim
                    }, os.path.join(experiment_dir, 'best_cossim_model.pth'))
                    print(f"  [+] Saved best sim model (Sim: {v_sim:.4f}).")
                    improved_this_epoch = True

                if (epoch + 1) % 1 == 0:
                    #torch.save(model_to_save.state_dict(), os.path.join(experiment_dir, f'epoch_{epoch + 1}.pth'))
                    torch.save({
                        'epoch': epoch,
                        'model': model_to_save.state_dict(),
                        'optim': optimizer.state_dict(),
                        'sim': v_sim
                    }, os.path.join(experiment_dir, f'epoch_{epoch + 1}.pth'))

                if improved_this_epoch:
                    counter = 0
                else:
                    counter += 1
                    print(f"  [!] No improvement for {counter}/{patience} epochs.")

                if counter >= patience:
                    print("\n[!] Early stopping triggered.")
                    stop_signal += 1

            if is_dist_avail_and_initialized():
                dist.all_reduce(stop_signal, op=dist.ReduceOp.SUM)

            if stop_signal.item() > 0:
                break

    except KeyboardInterrupt:
        if get_rank() == 0:
            print("\n[!] Training interrupted.")

    cleanup_ddp()

    if get_rank() == 0 and experiment_dir:
        print("\n[-] Training finished.")
        final_dir = f"{experiment_dir}_T{best_train_sim:.4f}_V{best_val_sim:.4f}_Resumed"
        try:
            os.rename(experiment_dir, final_dir)
            experiment_dir = final_dir
        except OSError:
            pass

        _run_plot_metrics(experiment_dir, conf)


if __name__ == '__main__':
    main()