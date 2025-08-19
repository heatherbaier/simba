import torch
import numpy as np
import pandas as pd
from haversine import haversine, Unit
from ..utils.utils import *
from ..data.instance_index import *
import torch.nn.functional as f


class Simba:
    def __init__(
        self,
        model,
        device = "cuda",
        baseline_image=None,
        baseline_coords=None,
        label = None,
        additional_model_inputs = None,
        coord_names=("lat", "lon"),
        distance_fn=None,
        perturbation_scheme="small_local"
    ):
        self.model = model.to(device)
        self.device = device
        self.baseline_image = baseline_image
        self.baseline_coords = baseline_coords
        self.label = label,
        self.coord_names = coord_names
        self.distance_fn = distance_fn or (lambda c1, c2: haversine(c1, c2, unit=Unit.KILOMETERS))
        self.perturbation_scheme = perturbation_scheme
        self.additional_model_inputs = additional_model_inputs

    def perturb_coordinates(self, coords, distance_km):
        """
        Generate perturbed coordinates at a given radial distance from coords.
        Uses small angular offsets based on the distance in km.
        """
        # print("COORDS: ", coords)
        lat, lon = coords
        offsets = [
            (lat + distance_km / 111.32, lon),  # north
            (lat - distance_km / 111.32, lon),  # south
            (lat, lon + distance_km / (111.32 * np.cos(np.radians(lat)))),  # east
            (lat, lon - distance_km / (111.32 * np.cos(np.radians(lat)))),  # west
        ]
        
        return offsets

    def get_gradients_wrt_inputs(self, image, coords):
        """
        Compute gradients of prediction wrt image pixels and coordinate inputs.
        """
        self.model.zero_grad()
        self.model.eval()

        import random

        # print("COORDS HERE: ", coords)
        
        image = image.to(self.device).unsqueeze(0).detach().requires_grad_(True) 
        coords = coords.to(self.device).detach().requires_grad_(True)

        # coords = torch.tensor([[ random.random(), random.random()]])
        # coords = coords.to(self.device).detach().requires_grad_(True)

        # print("COORDS HERE: ", coords)


        inputs = [image, coords]

        if self.additional_model_inputs is not None:
            for ami in self.additional_model_inputs:
                ami = ami.to(self.device).unsqueeze(0).detach().requires_grad_(True)
                inputs.append(ami)


        with torch.enable_grad():

            y = (self.model.net if hasattr(self.model, "net") else self.model)(*inputs)
            y_scalar = y.reshape(-1).mean()

            
            # y = self.model(*inputs)   # expect shape [1] or scalar
            # pick a scalar to differentiate; for regression, mean() is safe
            # y_scalar = y.reshape(-1).mean()

        
    
        # Gradients wrt coords and image
        g_image, g_coords = torch.autograd.grad(
            y_scalar, (image, coords),
            retain_graph=False,
            create_graph=False,
            allow_unused=True
        )

        # print("G coords: ", g_coords)
        # print("G image: ", g_image.shape)
        # print("Y: ", y)

        return g_image, g_coords

        # return image_grad, g_coords

    def compute_sensitivity(self, image, coords, normalize_by_distance=False, distance_km=None):
        """
        Calculate average gradient magnitudes for coords and image.
        """
        image_grad, coord_grad = self.get_gradients_wrt_inputs(image, coords)

        # print("CG: ", coord_grad, "\n")
        # print("IMG: ", image_grad)
        
        image_sens = image_grad.abs().mean().item()
        coord_sens = coord_grad.abs().mean().item()

        # print("image sens: ", image_sens)
        # print("coord sens: ", coord_sens)

        if normalize_by_distance and distance_km:
            coord_sens /= distance_km

        return {
            "image_sens": image_sens,
            "coord_sens": coord_sens,
            "normed_coord_sens": coord_sens if normalize_by_distance else None
        }

    def run_instance_analysis(self, distances_km):
        """
        Run perturbations for one baseline instance.
        """
        results = []
        for dist in distances_km:
            perturbed_coords = self.perturb_coordinates(self.baseline_coords, dist)
            # print("dist: ", dist, "     |      PC: ", perturbed_coords)
            for pcoords in perturbed_coords:
                coords_tensor = torch.tensor(pcoords, dtype=torch.float32).unsqueeze(0)
                sens = self.compute_sensitivity(self.baseline_image, coords_tensor,
                                                normalize_by_distance=True,
                                                distance_km=dist)
                sens["distance_km"] = dist
                results.append(sens)

        df = pd.DataFrame(results)
        df = df.groupby(df["distance_km"]).mean().reset_index()
        return df

    def run_global_analysis(self, sample_images, sample_coords, distances_km):
        """
        Aggregate sensitivity across many samples.
        """
        all_results = []
        for img, coords in zip(sample_images, sample_coords):
            self.baseline_image = img
            self.baseline_coords = coords
            df = self.run_instance_analysis(distances_km)
            all_results.append(df)
        return pd.concat(all_results, ignore_index=True)

    def plot_sensitivity_map(self, results_df):
        """
        Placeholder for mapping sensitivity (folium/geopandas/matplotlib).
        """
        raise NotImplementedError("Mapping not yet implemented.")

    def export_results(self, results_df, filepath):
        """
        Save results to CSV.
        """
        results_df.to_csv(filepath, index=False)
