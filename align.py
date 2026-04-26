import argparse
import logging
import time
from pathlib import Path

import cv2 as cv
import numpy as np
from image_utils import get_image_files, setup_logger

# FLANN matcher parameters
FLANN_TREES = 10
FLANN_CHECKS = 75

# Feature matching thresholds
LOWE_RATIO = 0.75
MIN_MATCH_COUNT = 4
RANSAC_THRESHOLD = 5.0
MIN_INLIER_RATIO = 0.3

# Homography determinant bounds (detect extreme transformations)
DET_MIN, DET_MAX = 0.2, 5.0


def extract_features(img: np.ndarray, detector: cv.SIFT) -> tuple[list[cv.KeyPoint], np.ndarray | None]:
    """
    Extract SIFT features from an image and apply RootSIFT normalization.
    
    Args:
        img: Input image (BGR format)
        detector: SIFT detector instance
        
    Returns:
        Tuple of (keypoints, descriptors). Descriptors is None if no keypoints found.
    """
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    keypoints, descriptors = detector.detectAndCompute(gray_img, None)
    
    if len(keypoints) == 0:
        return ([], None)
    
    descriptors /= (descriptors.sum(axis=1, keepdims=True) + np.finfo(float).eps)
    descriptors = np.sqrt(descriptors)
    
    return (keypoints, np.float32(descriptors))


def get_good_matches(
    matcher: cv.FlannBasedMatcher,
    des1: np.ndarray,
    des2: np.ndarray,
    ratio: float = LOWE_RATIO
) -> list[cv.DMatch]:
    """
    Perform approximate nearest neighbor matching and filter using Lowe's ratio test.
    
    Args:
        matcher: FLANN-based matcher instance
        des1: Descriptors from first image
        des2: Descriptors from second image
        ratio: Lowe's ratio threshold (default 0.75)
        
    Returns:
        List of good matches that passed the ratio test
    """
    matches = matcher.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def save_keypoints_visualization(
    img: np.ndarray,
    keypoints: list[cv.KeyPoint],
    output_path: Path,
    logger: logging.Logger
) -> None:
    """
    Save a visualization of keypoints on an image.
    
    Args:
        img: Input image
        keypoints: List of keypoints to visualize
        output_path: Path where the visualization will be saved
        logger: Logger instance for info messages
    """
    img_kp = cv.drawKeypoints(
        img, 
        keypoints, 
        None, 
        flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    cv.imwrite(str(output_path), img_kp)
    logger.info(f"Saved {len(keypoints)} keypoints to {output_path.name}")


def save_debug_visualizations(
    img_prev: np.ndarray,
    kp_prev: list[cv.KeyPoint],
    img_curr: np.ndarray,
    kp_curr: list[cv.KeyPoint],
    good_matches: list[cv.DMatch],
    mask: np.ndarray,
    debug_dir: Path,
    img_name_stem: str,
    logger: logging.Logger
) -> None:
    """
    Save debug visualizations showing matches and inlier keypoints.
    
    Args:
        img_prev: Previous (reference) image
        kp_prev: Keypoints from previous image
        img_curr: Current image (before alignment)
        kp_curr: Keypoints from current image
        good_matches: List of good matches
        mask: Inlier mask from RANSAC
        debug_dir: Directory to save debug images
        img_name_stem: Stem of the current image filename (for naming outputs)
        logger: Logger instance
    """
    # Extract only inlier matches
    inlier_matches = [good_matches[i] for i in range(len(mask)) if mask[i]]
    
    # Draw matches between images
    img_matches = cv.drawMatches(
        img_prev, kp_prev,
        img_curr, kp_curr,
        inlier_matches, None,
        flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
    )
    cv.imwrite(str(debug_dir / f"{img_name_stem}_matches.png"), img_matches)
    
    # Draw inlier keypoints on the current image
    inlier_kps = [kp_curr[good_matches[i].trainIdx] 
                  for i in range(len(mask)) if mask[i]]
    img_kp = cv.drawKeypoints(
        img_curr, 
        inlier_kps, 
        None, 
        flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    cv.imwrite(str(debug_dir / f"{img_name_stem}_keypoints.png"), img_kp)
    
    logger.info(f"Saved debug visualizations with {len(inlier_kps)} inliers for {img_name_stem}")


def align_frame(
    img_curr: np.ndarray,
    img_curr_path: Path,
    img_prev: np.ndarray,
    kp_prev: list[cv.KeyPoint],
    des_prev: np.ndarray,
    sift: cv.SIFT,
    flann: cv.FlannBasedMatcher,
    logger: logging.Logger,
    debug_dir: Path | None = None,
    img_prev_original: np.ndarray | None = None,
) -> tuple[np.ndarray, list[cv.KeyPoint], np.ndarray] | None:
    """
    Align current image to previous image using feature matching and homography.
    
    This function:
    1. Extracts features from the current image
    2. Matches features between current and previous images
    3. Computes a homography transformation using RANSAC
    4. Validates the transformation
    5. Warps the current image to align with the previous
    
    Args:
        img_curr: Current image to align
        img_curr_path: Path to current image (for logging/debugging)
        img_prev: Previous (reference) image (may be aligned from previous iteration)
        kp_prev: Keypoints from previous image
        des_prev: Descriptors from previous image
        sift: SIFT detector instance
        flann: FLANN matcher instance
        logger: Logger instance
        debug_dir: Optional directory to save debug visualizations
        img_prev_original: Optional original unaligned previous image for debug visualization.
                          If provided, will be shown in matches visualization instead of aligned version.
        
    Returns:
        Tuple of (aligned_image, keypoints, descriptors) on success, None on failure
    """
    # Extract features from current image
    kp_curr, des_curr = extract_features(img_curr, sift)

    # Validate descriptors
    if des_prev is None or des_curr is None:
        logger.warning(f"Missing descriptors: {img_curr_path.name}")
        return None

    # Match features
    good_matches = get_good_matches(flann, des_prev, des_curr)

    if len(good_matches) < MIN_MATCH_COUNT:
        logger.warning(
            f"Not enough matches ({len(good_matches)}/{MIN_MATCH_COUNT}): "
            f"{img_curr_path.name}"
        )
        return None

    # Prepare point correspondences for homography estimation
    src_pts = np.float32([kp_curr[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_prev[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Compute homography using RANSAC
    homography, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, RANSAC_THRESHOLD)
    
    # Validate homography
    if homography is None:
        logger.warning(f"Homography computation failed: {img_curr_path.name}")
        return None
    
    inlier_ratio = np.sum(mask) / len(mask)
    if inlier_ratio < MIN_INLIER_RATIO:
        logger.warning(
            f"Low inlier ratio ({inlier_ratio:.2%}): {img_curr_path.name}"
        )
        return None

    # Check transformation validity (detect extreme warps)
    det = np.linalg.det(homography[:2, :2])
    if not (DET_MIN < det < DET_MAX):
        logger.warning(
            f"Transformation too extreme (det={det:.2f}): {img_curr_path.name}"
        )
        return None

    # Apply transformation
    h, w = img_prev.shape[:2]
    aligned = cv.warpPerspective(img_curr, homography, (w, h))
    
    # Save debug visualizations if requested
    if debug_dir:
        # Use original unaligned image for visualization if provided
        img_prev_for_debug = img_prev_original if img_prev_original is not None else img_prev
        save_debug_visualizations(
            img_prev_for_debug, kp_prev,
            img_curr, kp_curr,
            good_matches, mask,
            debug_dir, img_curr_path.stem,
            logger
        )
    
    return aligned, kp_curr, des_curr


def main():
    parser = argparse.ArgumentParser(
        description="Align sequential images"
    )
    parser.add_argument(
        "location", 
        type=str, 
        help="Path to folder containing images"
    )
    parser.add_argument(
        "--start", 
        type=int, 
        default=1, 
        help="Starting image index (1-indexed)"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug mode and save visualization images"
    )
    args = parser.parse_args()

    # Setup
    logger = setup_logger("align", args.debug)
    files = get_image_files(args.location)
    
    if not files:
        logger.error("No valid images found in the specified location.")
        return
    
    logger.info(f"Found {len(files)} images to process")
    
    # Create output directories
    output_dir = Path(args.location + "_align")
    output_dir.mkdir(exist_ok=True)
    
    debug_dir = None
    if args.debug:
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(exist_ok=True)
        logger.info(f"Debug mode enabled. Visualizations will be saved to {debug_dir}")

    # Initialize FLANN matcher
    index_params = {'algorithm': 1, 'trees': FLANN_TREES}
    search_params = {'checks': FLANN_CHECKS}
    flann = cv.FlannBasedMatcher(index_params, search_params)

    # Load and process the reference (starting) image
    img_prev_path = files[args.start - 1]
    img_prev = cv.imread(str(img_prev_path))
    
    if img_prev is None:
        logger.error(f"Failed to load the starting image: {img_prev_path}")
        return

    logger.info(f"Using {img_prev_path.name} as reference image")
    
    # Extract features from reference image
    sift = cv.SIFT_create()
    kp_prev, des_prev = extract_features(img_prev, sift)
    
    if debug_dir:
        save_keypoints_visualization(
            img_prev, kp_prev,
            debug_dir / f"{img_prev_path.stem}_keypoints.png",
            logger
        )

    # Process each subsequent image
    successful_alignments = 0
    failed_alignments = 0
    img_prev_original = img_prev  # Keep track of original unaligned image
    
    for i in range(args.start, len(files)):
        start_time = time.perf_counter()
        img_curr_path = files[i]
        img_curr = cv.imread(str(img_curr_path))

        if img_curr is None:
            logger.warning(f"Failed to read: {img_curr_path.name}")
            failed_alignments += 1
            continue

        # Attempt alignment
        result = align_frame(
            img_curr, img_curr_path,
            img_prev, kp_prev, des_prev,
            sift, flann, logger, debug_dir,
            img_prev_original  # Pass original unaligned image for debug
        )

        if result is None:
            failed_alignments += 1
            continue

        # Update reference for next iteration
        aligned, kp_prev, des_prev = result
        img_prev = aligned
        img_prev_original = img_curr  # Next iteration's original is current aligned image

        # Save aligned image
        cv.imwrite(str(output_dir / img_curr_path.name), aligned)
        successful_alignments += 1
        
        elapsed = time.perf_counter() - start_time
        logger.info(f"Aligned {img_curr_path.name} in {elapsed:.3f}s")
        
        # Save debug visualization of aligned image keypoints
        if debug_dir:
            save_keypoints_visualization(
                img_curr, kp_prev,
                debug_dir / f"{img_curr_path.stem}_aligned_keypoints.png",
                logger
            )
    
    # Print summary
    total_processed = successful_alignments + failed_alignments
    logger.info("Alignment complete!")
    logger.info(f"Successful: {successful_alignments}/{total_processed}")
    logger.info(f"Failed: {failed_alignments}/{total_processed}")
    logger.info(f"Output: {output_dir}")
    

if __name__ == "__main__":
    main()
