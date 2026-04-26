import argparse

import cv2 as cv
import imageio
from image_utils import get_image_files

DEFAULT_FRAME_DURATION = 100

def create_gif(
    image_files,
    output_path,
    target_shape,
    frame_duration: int = DEFAULT_FRAME_DURATION,
) -> bool:
    """
    Create an animated GIF from a sequence of images.
    
    Args:
        image_files: List of Path objects to images in order
        output_path: Path where the GIF will be saved
        target_shape: desired w,h of GIF, (default: first frame)
        frame_duration: Duration of each frame in milliseconds (default: 100)
        
    Returns:
        True if successful, False otherwise
    """
    if not image_files:
        print("No images provided")
        return False
    
    # Load all images
    frames = []
    if target_shape:
        w,h = target_shape

    for i, img_path in enumerate(image_files):
        img = cv.imread(str(img_path))
        if img is None:
            print(f"Failed to load: {img_path.name}")
            continue
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        if target_shape is None:
            target_shape = img_rgb.shape
            h,w = target_shape[:2]
            
        if target_shape != img_rgb.shape:
            img_rgb = cv.resize(img_rgb, (w,h), interpolation=cv.INTER_LINEAR)

        frames.append(img_rgb)
        print(f"Loaded {i+1}/{len(image_files)}: {img_path.name}")


    if not frames:
        print("No images could be loaded")
        return False
    
    # Create GIF
    try:
        imageio.mimwrite(str(output_path), frames, duration=frame_duration, loop=0)
        print("Successfully created GIF")
        return True
    except Exception as e:
        print(f"Failed to create GIF: {e}")
        return False

def main():
    """Main entry point for GIF creation."""
    parser = argparse.ArgumentParser(
        description="Create an animated GIF from a sequence of images"
    )
    parser.add_argument(
        "location",
        type=str,
        help="Path to folder containing images (or output from align.py)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default = "output.gif",
        help="Output GIF filename (default: output.gif)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=DEFAULT_FRAME_DURATION,
        help=f"Frame duration in milliseconds (default: {DEFAULT_FRAME_DURATION})"
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="Frame shape in w h (default: First frame shape)"
    )
    args = parser.parse_args()

    image_files = get_image_files(args.location)

    if not image_files:
        print("No valid images found in the specified location.")
        return False
    
    print(f"Found {len(image_files)} images")
    
    # Create GIF
    create_gif(image_files, args.output, args.shape, args.duration)
    

if __name__ == "__main__":
    main()
