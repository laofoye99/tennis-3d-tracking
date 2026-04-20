"""Evaluation configuration for bounce detection framework."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalConfig:
    """Configuration for bounce detection evaluation."""

    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    gt_bounce_file: Path = field(default=None)
    cam66_dir: Path = field(default=None)
    cam68_dir: Path = field(default=None)
    homography_path: Path = field(default=None)

    # Camera positions (V2 coords — from config.yaml)
    cam66_pos: list[float] = field(default_factory=lambda: [0.165, -17.042, 6.217])
    cam68_pos: list[float] = field(default_factory=lambda: [0.211, 17.156, 5.286])

    # Evaluation params
    frame_tolerance: int = 5       # frames for bounce matching
    position_tolerance: float = 0.5  # meters for position matching
    max_frames: int = 3000

    # Detection params
    tracknet_model: str = "model_weight/TrackNet_finetuned.pt"
    median_bg_thresh: int = 10
    median_bg_block: int = 30

    # Model params
    input_size: tuple[int, int] = (288, 512)
    device: str = "cuda"
    tracknet_seq_len: int = 8
    heatmap_threshold: float = 0.3

    # Bounce detection params
    bounce_z_max: float = 0.5
    bounce_prominence: float = 0.10
    bounce_min_distance: int = 5
    bounce_smooth: int = 3

    # Tracker params (track-first approach)
    tracker_max_pixel_dist: int = 80
    tracker_max_gap: int = 3
    tracker_min_len: int = 10

    # Triangulation params
    max_ray_dist: float = 1.0
    min_overlap: int = 10

    def __post_init__(self):
        """Resolve default paths relative to project_root."""
        if self.gt_bounce_file is None:
            self.gt_bounce_file = Path("D:/tennis/blob_frame_different/bounce_results.json")
        if self.cam66_dir is None:
            self.cam66_dir = self.project_root / "uploads" / "cam66_20260307_173403_2min"
        if self.cam68_dir is None:
            self.cam68_dir = self.project_root / "uploads" / "cam68_20260307_173403_2min"
        if self.homography_path is None:
            self.homography_path = self.project_root / "src" / "homography_matrices.json"

    def validate(self) -> list[str]:
        """Check that all required paths exist. Returns list of errors."""
        errors = []
        if not self.gt_bounce_file.exists():
            errors.append(f"GT bounce file not found: {self.gt_bounce_file}")
        if not self.cam66_dir.exists():
            errors.append(f"cam66 directory not found: {self.cam66_dir}")
        if not self.cam68_dir.exists():
            errors.append(f"cam68 directory not found: {self.cam68_dir}")
        if not self.homography_path.exists():
            errors.append(f"Homography file not found: {self.homography_path}")
        model_path = self.project_root / self.tracknet_model
        if not model_path.exists():
            errors.append(f"TrackNet model not found: {model_path}")
        return errors
