import typer
import torch
from src.simba.core.registries import ModelRegistry, DatasetRegistry
from src.simba.training.trainer import Trainer, TrainConfig
from src.simba.explain import Simba
from src.simba.data.adapters import JSONGeoAdapter, resolve_json_paths
from torch.nn.functional import l1_loss



model = "tang2015"
model_ckpt = "artifacts/checkpoints/model.pt"
dataset = "json"
data_root = "/sciclone/geograd/heather_data/clean_dataset_jsons"
prefix = "phl"
with_neighbors = True
distances = "0.25,1,5"
index = 0
out_csv = "artifacts/sensitivity_instance.csv"


ys, coords, dup = resolve_json_paths(data_root, prefix, with_neighbors=with_neighbors)
ds = JSONGeoAdapter(root_dir=data_root, ys_path=ys, coords_path=coords, dup_path=dup, batch_size=8)

mw = ModelRegistry.get(model)()
mw.load(model_ckpt)
mw.net.to("cuda")

# Grab one sample from test set
sample = next(iter(ds.test_loader()))
img = sample["image"][index]
crd = sample["coords"][index]
nb  = sample.get("neighbor_images", None)
nm  = sample.get("neighbor_mask", None)
if nb is not None: nb = nb[index]
if nm is not None: nm = nm[index]

device = "cuda"

# Make the final vars leaf tensors that require grad
image  = img.to(device).unsqueeze(0).detach().requires_grad_(True)     # [1,3,H,W]
coords = crd.to(device).unsqueeze(0).detach().requires_grad_(True)     # [1,2]
nb     = nb.to(device).unsqueeze(0).detach().requires_grad_(True)      # [1,N,3,H,W]
nm     = nm.to(device).unsqueeze(0)   
# print(nb.shape)
# B, N, C, H, W = nb.shape
# nb_flat = nb.view(B * N, C, H, W)
# nb_flat = nb_flat.requires_grad_(True).to("cuda")
# nw_flat = nw_flat.requires_grad_(True).to("cuda")

# (If you still want .grad on non-leafs, you can do: image.retain_grad())

inputs = [image, coords, nb, nm]

out = mw.net(*inputs)

target = torch.tensor([[12.0]], device = "cuda")  # regression target shape [B,1]
# loss = torch.nn.functional.l1_loss(out, target)
loss = outut.mean()
loss.backward()

print(image.grad)  # now not None

