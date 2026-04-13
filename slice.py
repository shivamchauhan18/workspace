#!/usr/bin/env python3
"""
3D Slicer Image Segmentation Script

This script demonstrates how to use 3D Slicer programmatically for
medical image segmentation. It supports both the Slicer Python environment
and standalone usage via the 'slicer' package where applicable.

Usage:
    # Run within 3D Slicer:
    # Windows: "C:\\Program Files\\Slicer 5.x\\Slicer.exe" --no-splash --python-script slice.py
    # Linux: ./Slicer --no-splash --python-script slice.py
    # macOS: /Applications/Slicer.app/Contents/MacOS/Slicer --no-splash --python-script slice.py

    # Or using Slicer's Python:
    ./Slicer --python-script slice.py --input /path/to/image.nii.gz --output /path/to/output/
"""

import os
import sys
import argparse
from pathlib import Path


def setup_slicer_environment():
    """
    Configure the Slicer environment for segmentation tasks.
    Loads required modules and sets up the scene.
    """
    try:
        import slicer
        from slicer import ScriptedLoadableModule
    except ImportError:
        print("ERROR: This script must be run within 3D Slicer's Python environment.")
        print("Please run using: Slicer --python-script slice.py [arguments]")
        sys.exit(1)

    # Load required modules
    module_names = [
        'Volumes',
        'Segmentations',
        'SegmentEditor',
        'SegmentEditorEffect'
    ]

    for module_name in module_names:
        if not slicer.app.moduleManager().loadModule(module_name):
            print(f"Warning: Could not load module {module_name}")

    return slicer


def load_volume(file_path, slicer_module):
    """
    Load a medical image volume (NIfTI, DICOM, etc.) into Slicer.

    Args:
        file_path: Path to the image file
        slicer_module: The slicer module reference

    Returns:
        Volume node object
    """
    file_path = Path(file_path).resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"Image file not found: {file_path}")

    # Load the volume
    loaded = False
    if str(file_path).endswith('.nii') or str(file_path).endswith('.nii.gz'):
        volume_node = slicer_module.util.loadVolume(str(file_path))
        loaded = volume_node is not None
    elif str(file_path).endswith('.nrrd'):
        volume_node = slicer_module.util.loadVolume(str(file_path))
        loaded = volume_node is not None
    elif str(file_path).endswith('.mha') or str(file_path).endswith('.mhd'):
        volume_node = slicer_module.util.loadVolume(str(file_path))
        loaded = volume_node is not None
    else:
        # Try generic volume loading
        volume_node = slicer_module.util.loadVolume(str(file_path))
        loaded = volume_node is not None

    if not loaded:
        raise RuntimeError(f"Failed to load volume from {file_path}")

    print(f"Successfully loaded volume: {volume_node.GetName()}")
    return volume_node


def create_segmentation_node(name, slicer_module):
    """
    Create a new segmentation node.

    Args:
        name: Name for the segmentation
        slicer_module: The slicer module reference

    Returns:
        Segmentation node object
    """
    segmentation_node = slicer_module.mrmlScene.AddNewNodeByClass(
        "vtkMRMLSegmentationNode"
    )
    segmentation_node.SetName(name)
    return segmentation_node


def apply_threshold_segmentation(volume_node, segmentation_node,
                                  lower_threshold, upper_threshold,
                                  segment_name="Segment1",
                                  slicer_module=None):
    """
    Apply threshold-based segmentation to a volume.

    Args:
        volume_node: The input volume node
        segmentation_node: The segmentation node to store results
        lower_threshold: Lower intensity threshold
        upper_threshold: Upper intensity threshold
        segment_name: Name for the created segment
        slicer_module: The slicer module reference
    """
    # Create segment editor
    segment_editor_widget = slicer_module.qMRMLSegmentEditorWidget()
    segment_editor_widget.setMRMLScene(slicer_module.mrmlScene)
    segment_editor_node = slicer_module.mrmlScene.AddNewNodeByClass(
        "vtkMRMLSegmentEditorNode"
    )
    segment_editor_widget.setMRMLSegmentEditorNode(segment_editor_node)
    segment_editor_widget.setSegmentationNode(segmentation_node)
    segment_editor_widget.setMasterVolumeNode(volume_node)

    # Create a new segment
    segmentation_node.GetSegmentation().AddEmptySegment(segment_name)

    # Set active segment
    segment_editor_widget.setCurrentSegmentID(segment_name)

    # Apply threshold effect
    segment_editor_widget.setActiveEffectByName("Threshold")
    effect = segment_editor_widget.activeEffect()

    # Set threshold values
    effect.setParameter("MinimumThreshold", str(lower_threshold))
    effect.setParameter("MaximumThreshold", str(upper_threshold))

    # Apply the threshold
    effect.self().onApply()

    print(f"Applied threshold segmentation [{lower_threshold}, {upper_threshold}] to {segment_name}")


def apply_grow_from_seeds(volume_node, segmentation_node,
                          seed_points, slicer_module=None):
    """
    Apply 'Grow from Seeds' segmentation with user-defined seed points.

    Args:
        volume_node: The input volume node
        segmentation_node: The segmentation node
        seed_points: List of (x, y, z) seed coordinates
        slicer_module: The slicer module reference
    """
    segment_editor_widget = slicer_module.qMRMLSegmentEditorWidget()
    segment_editor_widget.setMRMLScene(slicer_module.mrmlScene)
    segment_editor_node = slicer_module.mrmlScene.AddNewNodeByClass(
        "vtkMRMLSegmentEditorNode"
    )
    segment_editor_widget.setMRMLSegmentEditorNode(segment_editor_node)
    segment_editor_widget.setSegmentationNode(segmentation_node)
    segment_editor_widget.setMasterVolumeNode(volume_node)

    # Create segments for each seed
    for i, point in enumerate(seed_points):
        segment_name = f"Seed_{i+1}"
        segmentation_node.GetSegmentation().AddEmptySegment(segment_name)

        segment_editor_widget.setCurrentSegmentID(segment_name)
        segment_editor_widget.setActiveEffectByName("Draw")

        # Place seed point (simplified - in practice use fiducials)
        print(f"Seed point {i+1}: {point}")

    segment_editor_widget.setActiveEffectByName("Grow from seeds")
    effect = segment_editor_widget.activeEffect()
    effect.self().onApply()


def apply_watershed_segmentation(volume_node, segmentation_node,
                                  slicer_module=None):
    """
    Apply watershed segmentation.

    Args:
        volume_node: The input volume node
        segmentation_node: The segmentation node
        slicer_module: The slicer module reference
    """
    # Create segment editor
    segment_editor_widget = slicer_module.qMRMLSegmentEditorWidget()
    segment_editor_widget.setMRMLScene(slicer_module.mrmlScene)
    segment_editor_node = slicer_module.mrmlScene.AddNewNodeByClass(
        "vtkMRMLSegmentEditorNode"
    )
    segment_editor_widget.setMRMLSegmentEditorNode(segment_editor_node)
    segment_editor_widget.setSegmentationNode(segmentation_node)
    segment_editor_widget.setMasterVolumeNode(volume_node)

    # Create segment
    segmentation_node.GetSegmentation().AddEmptySegment("Watershed_Segment")
    segment_editor_widget.setCurrentSegmentID("Watershed_Segment")

    # Apply watershed
    segment_editor_widget.setActiveEffectByName("Watershed")
    effect = segment_editor_widget.activeEffect()
    effect.self().onApply()


def apply_automated_segmentation(volume_node, segmentation_node,
                                  model_type="TotalSegmentator",
                                  slicer_module=None):
    """
    Apply AI/ML-based automated segmentation using extensions.

    Supported models (require installation of extensions):
    - TotalSegmentator: Full body organ segmentation
    - MONAILabel: Various pre-trained models
    - DeepLearningSegmenter

    Args:
        volume_node: The input volume node
        segmentation_node: The segmentation node
        model_type: Type of automated segmentation model
        slicer_module: The slicer module reference
    """
    print(f"Attempting automated segmentation with {model_type}...")

    # Check if extension is installed
    if model_type == "TotalSegmentator":
        try:
            import TotalSegmentator
            # Run TotalSegmentator
            logic = TotalSegmentator.TotalSegmentatorLogic()
            logic.setup()
            logic.getParameterNode().SetParameter("InputVolumeNodeId",
                                                  volume_node.GetID())
            logic.getParameterNode().SetParameter("OutputSegmentationNodeId",
                                                  segmentation_node.GetID())
            logic.runSegmentation()
        except ImportError:
            print("TotalSegmentator extension not installed.")
            print("Install via: Slicer extension manager -> TotalSegmentator")

    elif model_type == "MONAILabel":
        try:
            import MONAILabel
            # Configure and run MONAILabel
            logic = MONAILabel.MONAILabelLogic()
            # Additional configuration needed
        except ImportError:
            print("MONAILabel extension not installed.")
            print("Install via: Slicer extension manager -> MONAILabel")


def save_segmentation(segmentation_node, output_path, format_type="nifti"):
    """
    Export segmentation to file.

    Args:
        segmentation_node: The segmentation node to export
        output_path: Output file path
        format_type: Output format ('nifti', 'nrrd', 'stl', 'labelmap')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format_type in ['nifti', 'nii', 'nii.gz']:
        # Export as labelmap NIfTI
        labelmap_volume_node = slicer.modules.volumes.logic().CreateLabelVolumeFromSegmentation(
            slicer.mrmlScene, segmentation_node
        )
        slicer.util.saveNode(labelmap_volume_node, str(output_path))

    elif format_type == 'nrrd':
        labelmap_volume_node = slicer.modules.volumes.logic().CreateLabelVolumeFromSegmentation(
            slicer.mrmlScene, segmentation_node
        )
        slicer.util.saveNode(labelmap_volume_node, str(output_path))

    elif format_type == 'stl':
        # Export as STL mesh
        slicer.util.saveNode(segmentation_node, str(output_path))

    elif format_type == 'labelmap':
        # Export as labelmap volume
        labelmap_volume_node = slicer.modules.volumes.logic().CreateLabelVolumeFromSegmentation(
            slicer.mrmlScene, segmentation_node
        )
        slicer.util.saveNode(labelmap_volume_node, str(output_path))

    print(f"Segmentation saved to: {output_path}")


def run_batch_segmentation(input_dir, output_dir, method='threshold',
                           threshold_range=(50, 255)):
    """
    Process multiple images in batch mode.

    Args:
        input_dir: Directory containing input images
        output_dir: Directory for output segmentations
        method: Segmentation method ('threshold', 'watershed', 'automated')
        threshold_range: Tuple of (lower, upper) threshold values
    """
    slicer = setup_slicer_environment()

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    supported_extensions = ['.nii', '.nii.gz', '.nrrd', '.mha', '.mhd', '.dcm']

    for file_path in input_dir.iterdir():
        if file_path.suffix.lower() in supported_extensions or \
           str(file_path).endswith('.nii.gz'):
            print(f"\nProcessing: {file_path.name}")

            try:
                # Load volume
                volume_node = load_volume(file_path, slicer)

                # Create segmentation
                seg_name = f"{file_path.stem}_segmentation"
                segmentation_node = create_segmentation_node(seg_name, slicer)

                # Apply segmentation based on method
                if method == 'threshold':
                    apply_threshold_segmentation(
                        volume_node, segmentation_node,
                        threshold_range[0], threshold_range[1],
                        slicer_module=slicer
                    )
                elif method == 'watershed':
                    apply_watershed_segmentation(
                        volume_node, segmentation_node, slicer_module=slicer
                    )

                # Save result
                output_path = output_dir / f"{file_path.stem}_seg.nii.gz"
                save_segmentation(segmentation_node, output_path)

            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")
                continue


def interactive_segmentation_demo():
    """
    Run an interactive demonstration of segmentation capabilities.
    Requires 3D Slicer GUI environment.
    """
    slicer = setup_slicer_environment()

    print("\n=== 3D Slicer Interactive Segmentation Demo ===\n")
    print("This demo shows how to use Slicer's Python interface for segmentation.")
    print("In a real application, you would load your medical images here.\n")

    # Clear the scene
    slicer.mrmlScene.Clear(0)

    # Create a sample volume (for demonstration)
    # In practice, load your actual image file
    print("Creating sample volume for demonstration...")
    sample_volume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
    sample_volume.SetName("Sample_Volume")

    # Create segmentation node
    print("Creating segmentation node...")
    segmentation_node = create_segmentation_node("Demo_Segmentation", slicer)

    print("\nSegmentation node created successfully!")
    print("Use the functions in this script to:")
    print("  - Apply threshold segmentation")
    print("  - Apply watershed segmentation")
    print("  - Apply grow-from-seeds segmentation")
    print("  - Use AI models (TotalSegmentator, MONAILabel)")
    print("  - Export results in various formats")

    print("\nExample usage:")
    print("  apply_threshold_segmentation(volume_node, segmentation_node, 50, 255)")
    print("  save_segmentation(segmentation_node, 'output.nii.gz')")


def main():
    """Main entry point for the segmentation script."""
    parser = argparse.ArgumentParser(
        description='3D Slicer Image Segmentation Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single file threshold segmentation
    Slicer --python-script slice.py --input image.nii.gz --output seg.nii.gz --method threshold --lower 50 --upper 255

    # Batch processing
    Slicer --python-script slice.py --batch --input-dir ./images/ --output-dir ./segmentations/ --method threshold

    # Automated segmentation (requires TotalSegmentator extension)
    Slicer --python-script slice.py --input ct_scan.nii.gz --output organs.nii.gz --method automated --model TotalSegmentator
        """
    )

    parser.add_argument('--input', '-i', type=str, help='Input image file path')
    parser.add_argument('--output', '-o', type=str, help='Output segmentation file path')
    parser.add_argument('--input-dir', type=str, help='Input directory for batch processing')
    parser.add_argument('--output-dir', type=str, help='Output directory for batch processing')
    parser.add_argument('--batch', action='store_true', help='Enable batch processing mode')
    parser.add_argument('--method', type=str, default='threshold',
                        choices=['threshold', 'watershed', 'automated', 'seeds'],
                        help='Segmentation method to use')
    parser.add_argument('--lower', type=int, default=50, help='Lower threshold value')
    parser.add_argument('--upper', type=int, default=255, help='Upper threshold value')
    parser.add_argument('--model', type=str, default='TotalSegmentator',
                        help='Model name for automated segmentation')
    parser.add_argument('--format', type=str, default='nifti',
                        choices=['nifti', 'nrrd', 'stl', 'labelmap'],
                        help='Output format')
    parser.add_argument('--demo', action='store_true', help='Run interactive demo')

    args = parser.parse_args()

    if args.demo:
        interactive_segmentation_demo()
        return

    if args.batch:
        if not args.input_dir or not args.output_dir:
            parser.error("Batch mode requires --input-dir and --output-dir arguments")
        run_batch_segmentation(
            args.input_dir,
            args.output_dir,
            method=args.method,
            threshold_range=(args.lower, args.upper)
        )
        return

    if not args.input or not args.output:
        parser.error("Single file mode requires --input and --output arguments")

    # Setup Slicer environment
    slicer = setup_slicer_environment()

    # Load input volume
    volume_node = load_volume(args.input, slicer)

    # Create segmentation node
    segmentation_node = create_segmentation_node("Segmentation", slicer)

    # Apply selected segmentation method
    if args.method == 'threshold':
        apply_threshold_segmentation(
            volume_node, segmentation_node,
            args.lower, args.upper,
            slicer_module=slicer
        )
    elif args.method == 'watershed':
        apply_watershed_segmentation(volume_node, segmentation_node, slicer_module=slicer)
    elif args.method == 'automated':
        apply_automated_segmentation(
            volume_node, segmentation_node,
            model_type=args.model,
            slicer_module=slicer
        )

    # Save result
    save_segmentation(segmentation_node, args.output, format_type=args.format)

    print(f"\nSegmentation complete: {args.output}")


# Alternative: Standalone script for environments without full Slicer GUI
def standalone_segmentation_demo():
    """
    Standalone demonstration using Python packages for basic image processing.
    This is an alternative when full 3D Slicer is not available.

    Requires: numpy, scipy, nibabel, SimpleITK
    """
    try:
        import numpy as np
        import nibabel as nib
        from scipy import ndimage
        import SimpleITK as sitk

        print("\n=== Standalone Segmentation (No Slicer GUI) ===\n")

        def load_image(path):
            """Load a medical image using nibabel or SimpleITK."""
            if path.endswith('.nii') or path.endswith('.nii.gz'):
                img = nib.load(path)
                return img.get_fdata(), img.affine
            else:
                img = sitk.ReadImage(path)
                return sitk.GetArrayFromImage(img), None

        def threshold_segmentation(image, lower, upper):
            """Simple threshold-based segmentation."""
            return ((image >= lower) & (image <= upper)).astype(np.uint8)

        def watershed_segmentation(image):
            """Watershed segmentation."""
            # Compute distance transform
            distance = ndimage.distance_transform_edt(image)
            # Find local maxima
            local_maxi = ndimage.maximum_filter(distance, size=7) == distance
            # Marker labeling
            markers, _ = ndimage.label(local_maxi)
            # Watershed
            labels = ndimage.watershed_ift(-distance, markers)
            return labels

        def save_segmentation(seg_data, affine, output_path):
            """Save segmentation as NIfTI."""
            seg_img = nib.Nifti1Image(seg_data, affine)
            nib.save(seg_img, output_path)

        print("Use these functions for basic segmentation without Slicer:")
        print("  - load_image(path)")
        print("  - threshold_segmentation(image, lower, upper)")
        print("  - watershed_segmentation(image)")
        print("  - save_segmentation(seg_data, affine, output_path)")

    except ImportError as e:
        print(f"Standalone mode requires additional packages: {e}")
        print("Install with: pip install numpy scipy nibabel SimpleITK")


if __name__ == '__main__':
    # Check if running within Slicer or standalone
    try:
        import slicer
        main()
    except ImportError:
        print("=" * 60)
        print("NOT RUNNING IN 3D SLICER ENVIRONMENT")
        print("=" * 60)
        print("\n3D Slicer segmentation requires running within Slicer's Python.")
        print("Options:\n")
        print("1. Install 3D Slicer and run:")
        print("   ./Slicer --python-script slice.py [arguments]\n")
        print("2. For basic segmentation without Slicer, use the standalone")
        print("   functions with: pip install numpy scipy nibabel SimpleITK\n")
        print("Running standalone demo...")
        standalone_segmentation_demo()
