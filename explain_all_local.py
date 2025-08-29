from src.simba.explain import Simba
from src.simba.cli.main import *

import json


# python3 -m simba.cli.main explain-instance   --model r18_wc --ckpt-dir /home/hbaier/projects/simba/artifacts/phl_bs32_r18_v2/   --dataset json   --data-root /scratch/hbaier/data_jsons   --prefix phl --distances 0.25,1,5,10,15,20 --no-neighbors --input-coords 15.119554775325298,120.83289931426896


with open("/scratch/hbaier/data_jsons/phl_coords.json", "r") as f:
    phl_coords = json.load(f)

# i = 0
for imname,im_coords in phl_coords.items():
    coords_txt = str(im_coords[1]) + "," + str(im_coords[0])
    explain_instance(model = "biasField",
                     ckpt_dir = "/home/hbaier/projects/simba/artifacts/phl_bs64_biasField_v1/",
                     dataset = "json",
                     data_root = "/scratch/hbaier/data_jsons",
                     prefix = "phl",
                     distances = "0.25,1,5,10,15,20",
                     input_coords = coords_txt
                     )

    # i += 1

    # if i > 4:
    #     break




