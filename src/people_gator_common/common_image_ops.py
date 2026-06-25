"""
Common image operations for the face detection and alignment pipelines.

ALL METHOD SHOULD BE MULTIPROCESSING SAFE (no shared state, no reliance on in-memory caching that may not be shared across processes, etc.)

This module includes functions for:
- Decoding raw image bytes into PIL Images and tv_tensors.Images
- Cropping images based on bounding boxes from detection results
- Adjusting detection results to be relative to the original image after cropping
- Computing similarity transforms to align detected faces to a reference keypoint configuration
- Applying standard transformations for timm models
"""


from PIL import Image
import numpy as np
import math

from opentelemetry import trace

tracer = trace.get_tracer(__name__)


@tracer.start_as_current_span("crop_bounding_box")
def crop_bounding_box(
    img: Image.Image,
    bbox: tuple[int, int, int, int]
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Crop the image to the given bounding box and return the cropped image and the actual bbox used for cropping."""
    h, w = img.height, img.width
    x_min, y_min, x_max, y_max = bbox
    if not (0 <= x_min < x_max <= w and 0 <= y_min < y_max <= h):
        raise ValueError(f"Invalid bbox coordinates: {bbox} for image size {(w,h)}")
    x1 = max(0, int(x_min))
    y1 = max(0, int(y_min))
    x2 = min(w, int(x_max))
    y2 = min(h, int(y_max))
    crop = img.crop((x1, y1, x2, y2))
    return crop, (x1, y1, x2, y2)

@tracer.start_as_current_span("compute_similarity_transform_with_no_scale")
def compute_similarity_transform_with_no_scale(src: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute the similarity transform (rotation + translation) that aligns src to ref, without scaling.
    Scale is returned separately as a scalar factor.
    
    Args:
        src: Nx2 array of source keypoints
        ref: Nx2 array of reference keypoints
    Returns:
        T: 3x3 similarity transform matrix that maps src to ref (that ignores scale)
        T_inv: 3x3 inverse similarity transform matrix that maps ref to src (that ignores scale)
        scale: scalar scale factor from src to ref
    """
    src_homogeneous = np.hstack([src, np.ones((src.shape[0], 1))]) # [x, y, 1] x N
    ref_homogeneous = np.hstack([ref, np.ones((ref.shape[0], 1))]) # [x, y, 1] x N
    T_center = np.eye(3)
    T_center[2, :2] = -src.mean(axis=0)
    T_center_inv = np.eye(3)
    T_center_inv[2, :2] = src.mean(axis=0)
    T_outwards = np.eye(3)
    T_outwards[2, :2] = ref.mean(axis=0)
    T_outwards_inv = np.eye(3)
    T_outwards_inv[2, :2] = -ref.mean(axis=0)

    src_centered = src_homogeneous @ T_center # [x_centered, y_centered] x N
    ref_centered = ref_homogeneous @ T_outwards_inv # [x_centered, y_centered] x N
    scale = (((ref_centered[:, :2] ** 2).sum(1) ** 0.5)  / ((src_centered[:, :2] ** 2).sum(1) ** 0.5)).mean() # average scale factor across keypoints
    T_scale = np.eye(3)
    T_scale_inv = np.eye(3)
    T_scale[:2, :2] = np.eye(2) * scale
    T_scale_inv[:2, :2] = np.eye(2) / scale
    src_centered_scaled = src_centered @ T_scale
    U, _, V = np.linalg.svd(ref_centered[:, :2].T @ src_centered_scaled[:, :2])
    T_rot = np.eye(3)
    T_rot[:2, :2] = (U @ V).T
    T_rot_inv = np.eye(3)
    T_rot_inv[:2, :2] = np.linalg.inv(T_rot[:2, :2])

    T = T_center @ T_scale @ T_rot @ T_outwards @ T_scale_inv
    T_inv = T_scale @ T_outwards_inv @ T_rot_inv @ T_scale_inv @ T_center_inv
    return T, T_inv, scale

@tracer.start_as_current_span("align_crop_to_reference")
def align_crop_to_reference(
    image: Image.Image,
    source_kps: list[tuple[float, float]],
    reference_kps_normalized: list[tuple[float, float]],
) -> Image.Image:
    """Rotate and crop `face_img` so that `src_kps` align to `reference_kps_normalized`.
    
    - `reference_kps_normalized` are (x,y) pairs in normalized coordinates [0,1]
        relative to the face image (width, height).
    - If `reference_kps_normalized` contains fewer than 2 valid points, raises a ValueError.
    - The function computes a similarity transform (scale+rotation+translation)
    - It applies the transform to the image and then returns a cropped patch centered around the transformed keypoints to remove excess background.
    - For a single face
    """
    if len(reference_kps_normalized) < 2:
        raise ValueError("At least 2 reference keypoints are required for alignment")
    if len(source_kps) != len(reference_kps_normalized):
        raise ValueError("Number of source keypoints must match number of reference keypoints")
    
    _, T_inv, scale = compute_similarity_transform_with_no_scale(np.array(source_kps), np.array(reference_kps_normalized))
    image_size = (math.ceil(1/scale), math.ceil(1/scale))
    transformed = image.transform(
        image_size,
        Image.Transform.AFFINE,
        data=(T_inv[0, 0], T_inv[1, 0], T_inv[2, 0], T_inv[0, 1], T_inv[1, 1], T_inv[2, 1]),
        resample=Image.Resampling.BILINEAR,
    )
    return transformed
