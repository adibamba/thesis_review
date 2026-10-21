import time
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.optim as optim

app = FastAPI(title="Azure Hybrid Neuro-Vision Cloud Server")

# Fine-tuning classification head anchored to YOLO bottleneck dimension (256-d)
class CloudBackboneFineTuner(nn.Module):
    def __init__(self, in_features=256, num_classes=80):
        super().__init__()
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.classifier(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CloudBackboneFineTuner().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)  # Cloud Adam-BP
criterion = nn.CrossEntropyLoss()

class OffloadPayload(BaseModel):
    frame_id: int
    features: list  # Bottleneck tensor feature vector
    pseudo_label: int = 0
    free_energy: float

@app.get("/health")
def health_check():
    return {"status": "online", "engine": "Azure Cloud BP", "device": str(device)}

@app.post("/api/v1/cloud_bp_step")
def cloud_bp_step(payload: OffloadPayload):
    t_start = time.perf_counter()

    # Format payload tensor
    x = torch.tensor([payload.features], dtype=torch.float32, device=device)
    y = torch.tensor([payload.pseudo_label], dtype=torch.long, device=device)

    # Execute full backward pass (Cloud Backpropagation)
    optimizer.zero_grad()
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()

    compute_duration_ms = (time.perf_counter() - t_start) * 1000.0

    return {
        "status": "success",
        "frame_id": payload.frame_id,
        "loss": float(loss.item()),
        "cloud_compute_ms": round(compute_duration_ms, 2),
        "free_energy_received": payload.free_energy,
    }

# ---------------------------------------------------------------- additions for repeatable runs
class ResetPayload(BaseModel):
    seed: int = 0

@app.post("/api/v1/reset")
def reset(p: ResetPayload):
    """Re-create the model and optimizer from a fixed seed, so every server starts identically."""
    global model, optimizer
    torch.manual_seed(p.seed)
    model = CloudBackboneFineTuner().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    return {"status": "reset", "seed": p.seed, "weight_norm": float(model.classifier.weight.norm())}

@app.get("/api/v1/weights")
def weights():
    """Summary of the learned weights; call before and after a run to show they moved."""
    w = model.classifier.weight.detach().cpu().numpy()
    return {"shape": list(w.shape), "norm": float(np.linalg.norm(w)),
            "mean": float(w.mean()), "std": float(w.std()),
            "first_row_first_8": [round(float(v), 5) for v in w[0, :8]],
            "bias_norm": float(model.classifier.bias.detach().norm())}