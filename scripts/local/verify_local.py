"""Progress report for the local generation run.

Counts items with completed images and completed meshes (GLB + 3DGS PLY) across
all manifest geographies, and prints failure counts from state/.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompt_utils import MANIFESTS, PROJECT_DIR, load_manifest  # noqa: E402

OUTPUT_DIR = PROJECT_DIR / "outputs_relief"
STATE_DIR = PROJECT_DIR / "state"


def main():
    total_items = total_img = total_mesh = total_gs = 0
    for geo in MANIFESTS:
        items = load_manifest(geo)
        n_img = n_mesh = n_gs = 0
        for it in items:
            d = OUTPUT_DIR / geo / it["category"] / it["id"]
            if (d / f"{it['id']}.png").exists():
                n_img += 1
            if (d / f"{it['id']}.glb").exists() and (d / f"{it['id']}.glb").stat().st_size > 1024:
                n_mesh += 1
            if (d / f"{it['id']}.ply").exists():
                n_gs += 1
        total_items += len(items)
        total_img += n_img
        total_mesh += n_mesh
        total_gs += n_gs
        print(f"{geo:10s} {len(items):6d} items | images {n_img:6d} | "
              f"meshes {n_mesh:6d} | 3DGS {n_gs:6d}")

    print(f"{'TOTAL':10s} {total_items:6d} items | images {total_img:6d} | "
          f"meshes {total_mesh:6d} | 3DGS {total_gs:6d}")

    for name in ("local_image_failures.json", "local_mesh_failures.json"):
        p = STATE_DIR / name
        if p.exists():
            fails = json.loads(p.read_text())
            print(f"{name}: {len(fails)} failures")


if __name__ == "__main__":
    main()
