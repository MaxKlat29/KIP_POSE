# Synthetic Data Generation Pipeline for Isaac Sim

Automated pipeline for generating synthetic training data using NVIDIA Isaac Sim. Produces annotated images with bounding boxes, semantic segmentation, instance segmentation, and depth maps.

## Features

- **Two Spawn Modes**: Place objects in tray grid slots or spawn randomly with physics simulation
- **Multiple Asset Types**: Currently supports `Anker_Kurz` and `Anker_Lang` (easily extensible)
- **Rich Annotations**: RGB, 2D bounding boxes, semantic segmentation, instance segmentation, depth maps
- **Physics Simulation**: Auto-settling of objects (2 seconds / 120 frames) for realistic placement
- **Collision Detection**: Prevents object overlap in random mode
- **Deterministic Generation**: Seeded random number generator for reproducible datasets

## Usage Modes

### Tray Mode (`SPAWN_MODE = "tray"`)
Places objects in a grid-based tray system. Each object is randomly assigned a grid slot with optional Z-axis rotation.
- **Use case**: Organized bin/tray picking scenarios
- **Grid Configuration**: 7×5 slots with 4 excluded corners
- **Object Count**: Configurable via `NUM_ANKERS_IN_TRAY`

### Random Mode (`SPAWN_MODE = "random"`)
Spawns objects at random positions within defined bounds. Physics simulation runs for 120 frames (2 seconds) to let objects settle naturally.
- **Use case**: Loose packing/cluttered scenarios
- **Collision Detection**: Prevents interpenetration
- **Physics Settling**: 2-second simulation before capture (configurable via `PHYSICS_SETTLE_STEPS`)

## Running the Script

### In Isaac Sim Script Editor (Recommended for Development)
1. Open your scene in Isaac Sim
2. Open the **Script Editor** (Code > Script Editor)
3. Copy and paste the entire script or open the file
4. Click **Execute** (or use the play button)
5. Monitor output in the **Console** window

### Standalone Execution (Future)
The script is designed to eventually run as a standalone executable outside the Isaac Sim editor. Currently requires the Isaac Sim Python environment.

## Configuration

All configuration is located at the top of the script:

### Assets
```python
ASSETS = [
    {"path": "path/to/asset.usd", "label": "AssetName"},
    ...
]
```

### Output
```python
OUTPUT_DIR = "/path/to/output"
NUM_RENDERS = 2                 # Number of dataset samples to generate
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
```

### Spawn Settings
```python
SPAWN_MODE = "tray"             # or "random"
NUM_ANKERS_IN_TRAY = 5          # For tray mode
NUM_OBJECTS = 5                 # For random mode
PHYSICS_SETTLE_STEPS = 120      # Frames to simulate (2 seconds @ 60 FPS)
```

### Randomization
```python
RANDOM_SEED = 42                # Change for different variations, use None for fully random
```

## Output Format

For each render, the pipeline generates the following files:

### RGB Image
- **File**: `rgb_####.png`
- **Format**: Standard color image (RGB)
- **Resolution**: Configurable (default 1280×720)

### 2D Bounding Boxes
- **File**: `bbox_2d_####.json`
- **Format**: JSON with object-level bounding box annotations
- **Content**: Per-object tight bounding boxes in 2D image space

### Semantic Segmentation
- **File**: `semantic_####.png` + `semantic_labels_####.json`
- **Format**: Grayscale image with label → asset_name mapping
- **Labels**: Unique index per asset type (Anker_Kurz, Anker_Lang, etc.)

### Instance Segmentation
- **File**: `instance_####.png`
- **Format**: Grayscale image with unique ID per object instance
- **Use**: Training for instance detection/segmentation tasks

### Depth Map
- **File**: `depth_####.png`
- **Format**: Normalized 16-bit PNG
- **Encoding**: Distance from camera (normalized to 0-65535 range)

## Adding New Assets

To extend the pipeline with additional asset types:

1. **Prepare USD file**: Ensure your asset is a valid USD file with proper physics properties
2. **Add to configuration**:
   ```python
   ASSETS = [
       {"path": "path/to/Anker_Kurz.usd", "label": "Anker_Kurz"},
       {"path": "path/to/NewAsset.usd", "label": "NewAsset"},  # Add this
   ]
   ```
3. **Verify physics**: The script applies RigidBody and Collision APIs; ensure your asset doesn't conflict
4. **Test with small NUM_RENDERS**: Generate a few samples to verify correct appearance and physics

## Technical Details

### Physics Configuration
- **Linear Damping**: 0.1 (reduces sliding)
- **Angular Damping**: 0.1 (reduces spinning)
- **Collision Approximation**: Convex decomposition (accurate but performant)
- **Contact Offset**: 0.005 m

### Rotation Handling
- **Tray Mode**: Fixed base rotation (90°, 0°, 0°) + random Z-axis spin (0°-360°)
- **Random Mode**: Fully random orientation (0°-360° on all axes)

### Grid Layout (Tray Mode)
The tray grid uses specific spacing for physical accuracy:
- **Grid spacing X**: 0.03442 m
- **Grid spacing Y**: 0.033887 m
- **Excluded slots**: Four corners (0,0), (6,0), (0,4), (6,4)

## Future Enhancements

- [ ] Standalone script execution without Isaac Sim editor
- [ ] Support for additional object types and assets
- [ ] Configurable physics settling time per mode
- [ ] Lighting variation parameters
- [ ] Camera position/angle variations
- [ ] Automated dataset splitting (train/val/test)
- [ ] Integration with ML training pipelines

## Troubleshooting

**Camera not found error**: Verify the camera path matches your scene. The script will list valid cameras.

**Objects not settling in random mode**: Increase `PHYSICS_SETTLE_STEPS` or check collision configuration.

**Overlapping objects in random mode**: Decrease spawn bounds or increase `MAX_SPAWN_ATTEMPTS`.

**Poor semantic labels**: Ensure assets have proper semantic tags applied; the script auto-applies them via the asset label.

## Requirements

- NVIDIA Isaac Sim (tested with version 4.x)
- USD assets for objects
- Scene with configured camera at `ZIVID_CAMERA_PATH`
- Sufficient disk space for output images and annotations
