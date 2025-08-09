import typer
import torch
from ..core.registries import ModelRegistry, DatasetRegistry
from ..training.trainer import Trainer, TrainConfig
from ..explain.simba import Simba
from ..data.adapters import JSONGeoAdapter, resolve_json_paths


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


@app.command()
def explain_instance(
    model_ckpt: str,
    dataset: str = "json",
    data_root: str = typer.Option("", help="Folder with JSONs & images"),
    prefix: str = typer.Option("", help="Dataset prefix"),
    with_neighbors: bool = True,
    distances: str = "0.25,1,5",  # km
    index: int = 0,               # which sample from test set
    out_csv: str = "artifacts/sensitivity_instance.csv",
):
    # build dataset like in train
    if dataset == "json":
        from ..data.adapters import JSONGeoAdapter, resolve_json_paths
        ys, coords, dup = resolve_json_paths(data_root, prefix, with_neighbors=with_neighbors)
        ds = JSONGeoAdapter(root_dir=data_root, ys_path=ys, coords_path=coords, dup_path=dup, batch_size=8)
    else:
        ds = DatasetRegistry.get(dataset)()

    # load model
    mw = ModelRegistry.get("tang2015")()  # or model name param; for now, Tang
    mw.load(model_ckpt)

    # sample batch item
    sample = next(iter(ds.test_loader()))
    # choose one instance
    img = sample["image"][index]
    crd = sample["coords"][index]
    nb  = sample.get("neighbor_images", None)
    nm  = sample.get("neighbor_mask", None)
    if nb is not None: nb = nb[index]
    if nm is not None: nm = nm[index]

    analyzer = SpatialSensitivityAnalyzer(mw)
    distances_km = [float(x) for x in distances.split(",")]
    df = analyzer.run_instance_analysis(img, crd, distances_km, neighbor_images=nb, neighbor_mask=nm, regression=True)
    analyzer.export_results(df, out_csv)
    typer.echo(f"Saved instance sensitivity to {out_csv}")







if __name__ == "__main__":
    app()
