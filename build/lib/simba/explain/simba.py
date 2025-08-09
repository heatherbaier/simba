import torch
import numpy as np
import pandas as pd
from haversine import haversine, Unit

class Simba:
    def __init__(
        self,
        model,
        device,
        baseline_image=None,
        baseline_coords=None,
        coord_names=("lat", "lon"),
        distance_fn=None,
        perturbation_scheme="small_local"
    ):
        self.model = model.to(device)
        self.device = device
        self.baseline_image = baseline_image
        self.baseline_coords = baseline_coords
        self.coord_names = coord_names
        self.distance_fn = distance_fn or (lambda c1, c2: haversine(c1, c2, unit=Unit.KILOMETERS))
        self.perturbation_scheme = perturbation_scheme

    def perturb_coordinates(self, coords, distance_km):
        """
        Generate perturbed coordinates at a given radial distance from coords.
        Uses small angular offsets based on the distance in km.
        """
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
        self.model.eval()
        image = image.clone().detach().requires_grad_(True).to(self.device)
        coords = coords.clone().detach().requires_grad_(True).to(self.device)

        output = self.model(image, coords)
        loss = output.mean()  # For regression, mean prediction
        loss.backward()

        image_grad = image.grad.detach().cpu()
        coord_grad = coords.grad.detach().cpu()

        return image_grad, coord_grad

    def compute_sensitivity(self, image, coords, normalize_by_distance=False, distance_km=None):
        """
        Calculate average gradient magnitudes for coords and image.
        """
        image_grad, coord_grad = self.get_gradients_wrt_inputs(image, coords)
        image_sens = image_grad.abs().mean().item()
        coord_sens = coord_grad.abs().mean().item()

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
            for pcoords in perturbed_coords:
                coords_tensor = torch.tensor(pcoords, dtype=torch.float32).unsqueeze(0)
                sens = self.compute_sensitivity(self.baseline_image, coords_tensor,
                                                normalize_by_distance=True,
                                                distance_km=dist)
                sens["distance_km"] = dist
                results.append(sens)
        return pd.DataFrame(results)

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
