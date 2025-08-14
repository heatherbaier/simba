from pathlib import Path
import re

def highest_epoch(dir_path="."):
    pat = re.compile(r"^model_epoch(\d+)\.torch$")
    best = max(
        ((int(m.group(1)), p) for p in Path(dir_path).iterdir()
         if (m := pat.match(p.name))),
        default=(None, None),
    )
    return best  # (epoch_number, Path)

# epoch, path = highest_epoch("/path/to/your/folder")
# print(epoch, path)
