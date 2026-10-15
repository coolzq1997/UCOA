# Upper Crown Opening Angle (UCOA)

Code and sample data for the manuscript **"Upper Crown Opening Angle: A Phenotype-Driven Geometric Indicator for Treetop Dieback Detection from UAV-LiDAR"** (submitted to *Plant Phenomics*).

This repository implements treetop detection / individual-tree segmentation from UAV-LiDAR point clouds and computes three variants of the Crown/Apex Opening Angle (CAA) together with 12 conventional upper-crown metrics for treetop dieback assessment.

## Repository contents

```
.
├── UCOA.py            # Main script: segmentation, CAA calculation, figures and tables
├── requirements.txt   # Python dependencies
└── sampledata.txt     # Sample plot data (X, Y, Z point cloud; see format below)
```

## 1. Environment setup

The code is tested with Python 3.9–3.12 on Windows. We recommend a virtual environment:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

Dependencies: `numpy`, `scipy`, `scikit-image`, `matplotlib` (the `mpl_toolkits.mplot3d` module ships with Matplotlib).

## 2. Input data format

- The input is a plain-text point cloud file named **`sampledata.txt`**, placed in the same folder as `UCOA.py`.
- Each row is one point with **three whitespace-separated columns: `X  Y  Z`** (coordinates in **metres**). Additional columns, if present, are ignored.
- The point cloud is expected to contain the **upper-crown portion of the trees** (the upper part of the crowns used for the opening-angle analysis), not ground points.
- Points below the `min_height` threshold (default 2.0 m) are filtered out automatically.

Example (first rows):

```
3.25   8.11   12.47
3.31   8.06   12.52
3.28   8.15   12.41
```

A representative subset of plot data is provided in this repository for review. The full dataset is available from the corresponding author upon reasonable request.

## 3. Running

```bash
python UCOA.py
```

No command-line arguments are required. Key parameters (height threshold, CHM resolution, upper-crown analysis depth, scan-band width, etc.) are grouped in the "参数设置 / parameter settings" section at the top of the `if __name__ == '__main__':` block and can be edited directly.

## 4. Outputs

All results are written to a new folder **`CombinedResults_sampledata/`** in the working directory:

```
CombinedResults_sampledata/
├── 00_SegmentationOverview/
│   └── 00_CHM_segmentation_overview.png        # CHM, crown outer contours and detected treetops
├── 01_Tables/
│   ├── treetop_segmentation_parameters.txt     # Segmentation parameters used for this run
│   ├── CAA_angles_only.csv                     # Three CAA variants per tree
│   ├── CAA_diameter_details.csv                # EW/NS diameters, height differences and angles per method
│   └── CAA_and_12_conventional_upper_crown_metrics.csv  # CAA + 12 conventional metrics, one row per tree
├── 02_CAA_ComparisonFigures/
│   └── tree_001_CAA_comparison.png ...         # 2×3 comparison figure for each tree
└── 03_SingleMethodFigures_Optional/            # Not exported by default
    ├── CAA_fixed_section/
    ├── CAA_upper_max/
    └── CAA_constrained_apex_excluded/
```

The three CAA variants are:

| Column | Definition |
|---|---|
| `CAA_fixed_section_deg` | Opening angle from a fixed horizontal section 1 m below the treetop |
| `CAA_upper_max_deg` | Maximum opening angle within the 1 m upper-crown zone |
| `CAA_constrained_apex_excluded_deg` | Opening angle with a 0.2 m apex exclusion band and an endpoint height-difference constraint |

The 12 conventional metrics are: `Hmax_m`, `Hmean_m`, `CW_EW_m`, `CW_NS_m`, `CW_mean_m`, `CPA_m2`, `CV_hull_m3`, `CV_voxel_m3`, `PD_area_pts_m2`, `PD_volume_pts_m3`, `CanopyCover`, and `LAI_proxy` (a proxy only, not a field-measured LAI).

To additionally export single-method figures for every tree, set `export_single_method_figures = True` in the main block.

## Citation

Citation information will be added upon publication.

## Contact

Please contact the corresponding author for questions or full-data requests.
