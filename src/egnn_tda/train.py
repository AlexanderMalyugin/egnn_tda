import torch
from tqdm import tqdm

@torch.no_grad()
def compute_y_mean(train_loader, device):
    s, n = 0.0, 0
    for batch in train_loader:
        y = batch.y.view(-1).to(device)
        s += y.sum().item()
        n += y.numel()
    return torch.tensor(s / max(n, 1), device=device)

def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    epochs,
    device,
    print_every_epoch=10,
    ckpt_path="best_ckpt.pt",
    monitor="val",
):
    train_losses, val_losses = [], []
    best_loss = float("inf")
    global_step = 0

    pbar = tqdm(range(epochs), desc="Epoch", leave=True)
    y_mean = compute_y_mean(train_loader, device)

    for epoch in pbar:

        # -------- TRAIN --------
        model.train()
        sum_train_se, n_train = 0.0, 0

        for batch in train_loader:
            global_step += 1
            batch = batch.to(device)

            optimizer.zero_grad(set_to_none=True)

            output = model(batch).view(-1)
            y = batch.y.view(-1).to(output.dtype) - y_mean.to(output.dtype)

            se = (output - y).pow(2).sum()
            loss = se / y.numel()

            loss.backward()
            optimizer.step()

            sum_train_se += se.item()
            n_train += y.numel()

        train_mse = sum_train_se / max(n_train, 1)
        train_losses.append(train_mse)

        # -------- VAL --------
        model.eval()
        sum_val_se, n_val = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                output = model(batch).view(-1)
                y = batch.y.view(-1).to(output.dtype) - y_mean.to(output.dtype)

                se = (output - y).pow(2).sum()
                sum_val_se += se.item()
                n_val += y.numel()

        val_mse = sum_val_se / max(n_val, 1)
        val_losses.append(val_mse)

        if scheduler is not None:
            scheduler.step()

        # -------- BEST CHECKPOINT --------
        current = val_mse if monitor == "val" else train_mse
        if current < best_loss:
            best_loss = current
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_loss": best_loss,
                    "monitor": monitor,
                    "mean_y": y_mean,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
                    "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                },
                ckpt_path,
            )

        lr = optimizer.param_groups[0]["lr"] if optimizer is not None else float("nan")
        pbar.set_postfix(train_mse=f"{train_mse:.4e}", val_mse=f"{val_mse:.4e}", lr=f"{lr:.2e}")

        if (epoch + 1) % print_every_epoch == 0:
            tqdm.write(
                f"[epoch {epoch+1:03d}/{epochs:03d}] "
                f"train_mse={train_mse:.6f} val_mse={val_mse:.6f} lr={lr:.3e}"
            )

    return train_losses, val_losses, y_mean
