"""Quick script to inspect scans across IDs and verify orientation data is present."""

import numpy as np
from pathlib import Path

scenario_dir = Path("lidar_dataset_unspoofed/scenario_0000_forward_scan")
scan_files = sorted(scenario_dir.glob("scan_*.npz"))

# Print keys once
data0 = np.load(scan_files[0])
print("=" * 70)
print(f"Keys in each scan: {list(data0.keys())}")
print("=" * 70)

print(f"\n{'ID':>4}  {'Time':>7}  {'Position (x,y,z)':^32}  {'Quat (w,x,y,z)':^38}  {'Euler deg (r,p,y)':^30}")
print("-" * 120)

for scan_file in scan_files:
    data = np.load(scan_file)
    sid = int(data['scan_id'])
    t = float(data['timestamp'])
    pos = data['drone_position']
    q = data['drone_orientation_quat']
    euler_deg = np.degrees(data['drone_orientation_euler'])
    print(f"{sid:>4}  {t:>7.2f}  ({pos[0]:7.2f},{pos[1]:7.2f},{pos[2]:7.2f})  ({q[0]:.3f},{q[1]:.3f},{q[2]:.3f},{q[3]:.3f})  ({euler_deg[0]:6.1f},{euler_deg[1]:6.1f},{euler_deg[2]:6.1f})")
