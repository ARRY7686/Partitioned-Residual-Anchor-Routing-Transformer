"""Pipeline check: a pretrained ResNet-18 must memorise 64 DFG training frames (mAP on the same frames)."""
import os
os.environ.update(RART_DATASET="dfg", RART_KIND="resnet", RART_PRETRAINED="1", RART_SMOKE="1", RART_BATCH="16",
                  RART_DATA_DIR="/root/exp07/data", RART_OUT_DIR="/root/exp07/diag", RART_TAG="overfit_dfg")
src = open("/root/exp07/train.py").read()
exec(src[:src.index("def run():")])
torch.manual_seed(0)
model = build_model().to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
for step in range(30):
    for x, y, valid, _ in data.batches("train", train=True, generator=torch.Generator().manual_seed(step)):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = loss_fn(model(x), y, valid)
        opt.zero_grad(); loss.backward(); opt.step()
    if step % 10 == 9:
        model.eval(); scores, targets, valids = [], [], []
        with torch.no_grad():
            for x, y, valid, _ in data.batches("train", train=False):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    scores.append(model(x).float().cpu()); targets.append(y.cpu()); valids.append(valid.cpu())
        model.train()
        print(f"pass {step + 1}: loss {loss.item():.4f} | mAP on the same 64 frames "
              f"{average_precision(torch.cat(scores), torch.cat(targets), torch.cat(valids)):.1f}", flush=True)
