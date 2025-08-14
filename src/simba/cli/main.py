import typer
import torch, tqdm, os
import pandas as pd
from ..core.registries import ModelRegistry, DatasetRegistry
from ..training.trainer import Trainer, TrainConfig
from ..explain import Simba
from ..data.adapters import SimbaJSONDataset, JSONGeoAdapter, resolve_json_paths
from ..utils.utils import *


app = typer.Typer(help="SIMBA CLI")

@app.command()
def train(
    model: str = "tang2015",
    dataset: str = "json",   # 'json' triggers generic loader
    data_root: str = typer.Option("", help="Folder containing JSONs and images"),
    prefix: str = typer.Option("", help="Dataset prefix, e.g., 'phl' or 'western_africa'"),
    with_neighbors: bool = True,
    batch_size: int = 16,
    epochs: int = 2,
    lr: float = 1e-3,
    ckpt_dir: str = "artifacts/checkpoints",
):
    # Build dataset
    if dataset == "json":
        if not data_root or not prefix:
            raise typer.BadParameter("When dataset='json', you must pass --data-root and --prefix.")
        ys, coords, dup = resolve_json_paths(data_root, prefix, with_neighbors=with_neighbors)
        ds = JSONGeoAdapter(
            root_dir=data_root,
            ys_path=ys,
            coords_path=coords,
            dup_path=dup,
            batch_size=batch_size,
            ckpt_dir = ckpt_dir
        )
    else:
        ds = DatasetRegistry.get(dataset)()

    # Build model
    mw = ModelRegistry.get(model)()

    # Fail early if Tang model requires neighbors but user disabled it
    if model == "tang2015" and not with_neighbors:
        raise typer.BadParameter("Tang2015 model requires neighbors. Use --with-neighbors true.")

    cfg = TrainConfig(epochs=epochs, lr=lr, ckpt_dir=ckpt_dir)
    Trainer(mw, ds, cfg).fit()



@app.command()
def validate(model: str = "tang2015",
             dataset: str = "json",   # 'json' triggers generic loader
             data_root: str = typer.Option("", help="Folder containing JSONs and images"),
             prefix: str = typer.Option("", help="Dataset prefix, e.g., 'phl' or 'western_africa'"),
             with_neighbors: bool = True,
             ckpt_dir: str = "artifacts/checkpoints",
             device = "cuda"):

    mw = ModelRegistry.get(model)()#.build()

    epoch, path = highest_epoch(ckpt_dir)
    print(epoch, path)

    
    mw.load(path)
    # Build dataset

    # dsga
    

    if dataset == "json":
        
        if not data_root or not prefix:
            raise typer.BadParameter("When dataset='json', you must pass --data-root and --prefix.")
        ys, coords, dup = resolve_json_paths(data_root, prefix, with_neighbors=with_neighbors)
        ds = SimbaJSONDataset(
            root_dir=data_root,
            ys_path=ys,
            coords_path=coords,
            dup_path=dup,
            ckpt_dir = ckpt_dir,
            validate = True,
            
        )
    else:
        ds = DatasetRegistry.get(dataset)()


    mw.net = mw.net.to(device).eval()
    

    imnames, preds, labels = [], [], []
    for c, batch in tqdm.tqdm(enumerate(ds), desc = "Validating"):

        batch = {k: (v.to(device).unsqueeze(0) if hasattr(v, "to") else v) for k,v in batch.items()}

        out = mw.forward(batch)
        
        # print(, batch["label"])

        imnames.append(batch["image_name"])
        preds.append(out.item())
        labels.append(batch["label"].item())

        # print(imnames, preds, labels)
        
        
        if c % 10:
            df = pd.DataFrame()
            df["name"], df["pred"], df["label"] = imnames, preds, labels
            df.to_csv(os.path.join(ckpt_dir, f"epoch{epoch}_preds.csv"))

        df = pd.DataFrame()
        df["name"], df["pred"], df["label"] = imnames, preds, labels
        df.to_csv(os.path.join(ckpt_dir, f"epoch{epoch}_preds.csv"))  

        # print(c, len(ds), end = "\r")
        
        
        # print(i)
        # break


@app.command()
def explain(model_ckpt: str = "artifacts/checkpoints/model.pt",
           dataset: str = "toygeo",
           delta_small: int = 250, delta_large: int = 1000,
           n_batches: int = 5):
    # load model & data
    mw = ModelRegistry.get("from_ckpt")()
    mw.load(model_ckpt)
    ds = DatasetRegistry.get(dataset)()
    expl = SimbaExplainer(mw, ds)
    res = expl.global_sensitivity(deltas_m=(delta_small, delta_large), n_batches=n_batches)
    typer.echo(res)


@app.command("explain-instance")  # hyphenated CLI name
def explain_instance(
    model: str = "tang2015",
    model_ckpt: str = typer.Option(..., "--model-ckpt", help="Path to trained checkpoint"),
    dataset: str = "json",
    data_root: str = typer.Option("", help="Folder with JSONs & images"),
    prefix: str = typer.Option("", help="Dataset prefix, e.g., 'phl'"),
    with_neighbors: bool = True,
    distances: str = "0.25,1,5",
    index: int = 0,
    out_csv: str = "artifacts/sensitivity_instance.csv",
):
    # Build dataset (generic JSON mode or registered)
    if dataset == "json":
        if not data_root or not prefix:
            raise typer.BadParameter("When dataset='json', pass --data-root and --prefix.")
        ys, coords, dup = resolve_json_paths(data_root, prefix, with_neighbors=with_neighbors)
        ds = JSONGeoAdapter(root_dir=data_root, ys_path=ys, coords_path=coords, dup_path=dup, batch_size=8)
    else:
        ds = DatasetRegistry.get(dataset)()

    # Load model
    mw = ModelRegistry.get(model)()



    
    
    mw.load(model_ckpt)

    print(mw)

    # Grab one sample from test set
    sample = next(iter(ds.test_loader()))
    img = sample["image"][index]
    crd = sample["coords"][index]
    nb  = sample.get("neighbor_images", None)
    nm  = sample.get("neighbor_mask", None)
    if nb is not None: nb = nb[index]
    if nm is not None: nm = nm[index]

    # Run analyzer
    analyzer = Simba(model = mw.net,
                     baseline_image = img,
                     baseline_coords = crd,
                     additional_model_inputs = [nb, nm] # COME BACK AND MAKE THIS MORE FELXIBLE TO THE MODEL LATER ON!!!!
                    )  # uses GPU if available
    distances_km = [float(x) for x in distances.split(",")]
    df = analyzer.run_instance_analysis(
        # baseline_image=img,
        # baseline_coords=crd,
        distances_km=distances_km,
        # neighbor_images=nb,
        # neighbor_mask=nm,
        # regression=True,  # your current models are regression
    )
    analyzer.export_results(df, out_csv)
    typer.echo(f"Saved instance sensitivity to {out_csv}")


        # self,
        # model,
        # device = "cuda",
        # baseline_image=None,
        # baseline_coords=None,
        # coord_names=("lat", "lon"),
        # distance_fn=None,
        # perturbation_scheme="small_local"





if __name__ == "__main__":
    app()
