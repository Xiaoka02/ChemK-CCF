# -*- coding: utf-8 -*-
import os
import logging
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from torchvision import transforms
from tqdm import tqdm

# ImageMol configuration
IMAGEMOL_CONFIG = {
    'size': 224,
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
    'model': 'resnet50'
}


class ImageDataset(Dataset):
    def __init__(self, filenames, labels, index=None, img_transformer=None, normalize=None, ret_index=False, args=None):
        """
        Initialize image dataset
        Args:
            filenames: List of image file paths
            labels: List of labels
            index: Index list (optional)
            img_transformer: Image transformer
            normalize: Normalization function
            ret_index: Whether to return index
            args: Arguments object
        """
        super().__init__()
        self.args = args
        self.filenames = filenames
        self.labels = labels
        self.total = len(self.filenames)
        self.normalize = normalize
        self._image_transformer = img_transformer
        self.ret_index = ret_index

        # Add image transform
        self.transform = transforms.Compose([
            transforms.Resize((IMAGEMOL_CONFIG['size'], IMAGEMOL_CONFIG['size'])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGEMOL_CONFIG['mean'],
                std=IMAGEMOL_CONFIG['std']
            )
        ])

        # Check file existence
        if not all(os.path.exists(f) for f in filenames):
            missing_files = [f for f in filenames if not os.path.exists(f)]
            raise FileNotFoundError(f"Missing image files: {missing_files[:5]}...")

        # Handle index
        if index is not None:
            self.index = index
        else:
            self.index = [os.path.splitext(os.path.split(f)[1])[0] for f in filenames]

    def get_image(self, index):
        """
        Get image at specified index
        Args:
            index: Image index
        Returns:
            Processed image tensor
        """
        try:
            filename = self.filenames[index]
            img = Image.open(filename).convert('RGB')
            if self._image_transformer:
                img = self._image_transformer(img)
            return img
        except Exception as e:
            logging.error(f"Error loading image {filename}: {e}")
            # Return blank image
            blank_img = Image.new('RGB', (IMAGEMOL_CONFIG['size'], IMAGEMOL_CONFIG['size']), 'white')
            return self._image_transformer(blank_img) if self._image_transformer else blank_img

    def __getitem__(self, index):
        """
        Get dataset item
        Args:
            index: Index
        Returns:
            Data tuple (image, label) or (image, label, index)
        """
        data = self.get_image(index)
        if self.normalize is not None:
            data = self.normalize(data)
        if self.ret_index:
            return data, self.labels[index], self.index[index]
        return data, self.labels[index]

    def __len__(self):
        """Return dataset size"""
        return self.total


def Smiles2Img(smiles, size=(224, 224), savePath=None, quality=95):
    """
    Convert SMILES to molecule image
    Args:
        smiles: SMILES string
        size: Image size, can be integer or tuple
        savePath: Save path
        quality: Image quality
    Returns:
        PIL.Image object or None
    """
    try:
        # Handle size parameter
        if isinstance(size, int):
            size = (size, size)

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Generate 2D coordinates
        AllChem.Compute2DCoords(mol)

        # Use Draw.MolToImage directly
        img = Draw.MolToImage(
            mol,
            size=size,
            kekulize=True,
            wedgeBonds=True,
            fitImage=True
        )

        # Save image
        if savePath:
            img.save(savePath, quality=quality)

        return img

    except Exception as e:
        logging.error(f"Error generating image for SMILES {smiles}: {e}")
        return None


def load_or_generate_images(args, smiles):
    """
    Load or generate molecule images
    Supports batch processing, error recovery and retry mechanism
    """
    dataset_path = os.path.join('./data', args.dataset)
    image_folder = os.path.join(dataset_path, 'image')

    # Create necessary directories
    os.makedirs(image_folder, exist_ok=True)

    # Set default values
    regenerate_images = getattr(args, 'regenerate_images', False)
    image_size = getattr(args, 'image_size', IMAGEMOL_CONFIG['size'])
    image_quality = getattr(args, 'image_quality', 95)
    batch_size = getattr(args, 'image_batch_size', 1000)  # Batch processing size

    # Ensure image_size is a tuple
    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    total_generated = 0
    total_failed = 0
    failed_smiles = []  # Track failed SMILES for retry

    logging.debug(f"Starting to process {len(smiles)} molecules, batch size: {batch_size}")

    # First pass: process all molecules in batches
    for batch_start in range(0, len(smiles), batch_size):
        batch_end = min(batch_start + batch_size, len(smiles))
        batch_smiles = smiles[batch_start:batch_end]

        logging.debug(f"Processing batch {batch_start//batch_size + 1}: molecules {batch_start + 1}-{batch_end}")

        batch_generated = 0
        batch_failed = 0

        for idx_in_batch, smi in enumerate(tqdm(batch_smiles, desc=f"Batch {batch_start//batch_size + 1}", leave=False)):
            global_idx = batch_start + idx_in_batch
            save_path = os.path.join(image_folder, f'{global_idx + 1}.png')

            # If file doesn't exist or needs regeneration
            if not os.path.exists(save_path) or regenerate_images:
                try:
                    img = Smiles2Img(
                        smi,
                        size=image_size,
                        savePath=save_path,
                        quality=image_quality
                    )
                    if img is not None:
                        batch_generated += 1
                    else:
                        batch_failed += 1
                        failed_smiles.append((global_idx, smi))
                        logging.warning(f"Image generation failed: molecule {global_idx + 1}")
                except Exception as e:
                    batch_failed += 1
                    failed_smiles.append((global_idx, smi))
                    logging.error(f"Image generation error: molecule {global_idx + 1}, error: {str(e)}")

        total_generated += batch_generated
        total_failed += batch_failed

        logging.debug(f"Batch complete: generated {batch_generated}, failed {batch_failed}")

        # Add short delay between batches to avoid system overload
        import time
        time.sleep(0.1)

    # Second pass: retry failed molecules (up to 2 times)
    if failed_smiles:
        logging.debug(f"Starting retry for {len(failed_smiles)} failed molecules...")

        for retry_round in range(2):  # Up to 2 retry rounds
            if not failed_smiles:
                break

            logging.debug(f"Retry round {retry_round + 1}: remaining {len(failed_smiles)} molecules")

            still_failed = []
            retry_success = 0

            for global_idx, smi in failed_smiles:
                save_path = os.path.join(image_folder, f'{global_idx + 1}.png')

                try:
                    # Try with smaller image size on retry
                    retry_size = (image_size[0] // 2, image_size[1] // 2) if retry_round > 0 else image_size

                    img = Smiles2Img(
                        smi,
                        size=retry_size,
                        savePath=save_path,
                        quality=max(50, image_quality - 20 * retry_round)  # Reduce quality on retry
                    )

                    if img is not None:
                        total_generated += 1
                        total_failed -= 1
                        retry_success += 1
                        logging.debug(f"Retry success: molecule {global_idx + 1}")
                    else:
                        still_failed.append((global_idx, smi))
                        logging.warning(f"Retry still failed: molecule {global_idx + 1}")

                except Exception as e:
                    still_failed.append((global_idx, smi))
                    logging.error(f"Retry error: molecule {global_idx + 1}, {str(e)}")

            failed_smiles = still_failed
            if retry_success > 0:
                logging.debug(f"Retry round {retry_round + 1} success: {retry_success} molecules")

            # Short delay between retries
            time.sleep(0.5)

    if failed_smiles:
        logging.warning(f"Final number of failed molecules: {len(failed_smiles)}")
        logging.warning("Failed molecule indices: " + ", ".join([str(idx) for idx, _ in failed_smiles[:10]]) +
                       ("..." if len(failed_smiles) > 10 else ""))
    logging.debug("="*60)

    # Create placeholder images for final failed molecules
    if failed_smiles:
        for global_idx, smi in failed_smiles:
            save_path = os.path.join(image_folder, f'{global_idx + 1}.png')
            try:
                create_placeholder_image(save_path, image_size, f'Failed: {smi[:20]}...')
            except Exception as e:
                logging.error(f"Failed to create placeholder image: molecule {global_idx + 1}, {str(e)}")

        # Update statistics
        total_generated += len(failed_smiles)
        total_failed -= len(failed_smiles)

    return total_generated, total_failed


def create_placeholder_image(save_path, image_size, text="Invalid Molecule"):
    """
    Create placeholder image for failed molecules
    Args:
        save_path: Save path
        image_size: Image size (width, height)
        text: Display text
    """
    try:
        from PIL import Image, ImageDraw, ImageFont

        # Create white background image
        img = Image.new('RGB', image_size, color='white')
        draw = ImageDraw.Draw(img)

        # Try to load system font, fallback to basic font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()

        # Draw text in center of image
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        x = (image_size[0] - text_width) // 2
        y = (image_size[1] - text_height) // 2

        # Draw black border
        draw.rectangle([10, 10, image_size[0]-10, image_size[1]-10], outline='red', width=3)

        # Draw text
        draw.text((x, y), text, fill='black', font=font)

        # Save image
        img.save(save_path, 'PNG')

    except ImportError:
        # If PIL is not available, create simple numpy array image
        import numpy as np

        # Create white background
        img_array = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 255

        # Draw red X in center
        center_x, center_y = image_size[0] // 2, image_size[1] // 2
        half_size = min(image_size) // 4

        # Draw X
        for i in range(-half_size, half_size):
            if abs(i) < 3:  # Line width
                # Top-left to bottom-right
                x1 = center_x + i
                y1 = center_y + i
                x2 = center_x - i
                y2 = center_y - i

                if 0 <= x1 < image_size[0] and 0 <= y1 < image_size[1]:
                    img_array[y1, x1] = [255, 0, 0]  # Red
                if 0 <= x2 < image_size[0] and 0 <= y2 < image_size[1]:
                    img_array[y2, x2] = [255, 0, 0]  # Red

        # Save as PNG
        img = Image.fromarray(img_array)
        img.save(save_path, 'PNG')


def load_filenames_and_labels(args, image_folder, image_labels_csv):
    """
    Load image filenames and corresponding labels
    Args:
        args: Arguments object
        image_folder: Image folder path
        image_labels_csv: Labels file path
    Returns:
        tuple: (List of image file paths, labels array)
    """
    assert args.task_type in ["class", "reg"], f"Unsupported task type: {args.task_type}"

    try:
        # Check if files exist
        if not os.path.exists(image_labels_csv):
            raise FileNotFoundError(f"Labels file not found: {image_labels_csv}")
        if not os.path.exists(image_folder):
            raise FileNotFoundError(f"Image folder not found: {image_folder}")

        # Read CSV file
        df = pd.read_csv(image_labels_csv)

        # Check required columns
        if "filename" not in df.columns:
            raise ValueError("CSV file must contain 'filename' column")

        # Get filenames and labels
        filenames = df["filename"].values

        # Dynamically get label columns (all columns except filename)
        label_columns = [col for col in df.columns if col != "filename"]
        if not label_columns:
            raise ValueError("No label columns found in CSV file")

        labels = df[label_columns].values

        # Convert labels based on task type
        if args.task_type == "class":
            labels = labels.astype(int)
        else:  # reg
            labels = labels.astype(float)

        # Create full file paths
        full_paths = [os.path.join(image_folder, filename) for filename in filenames]

        # Verify files exist (all should exist now including placeholder images)
        missing_files = [f for f in full_paths if not os.path.exists(f)]
        if missing_files:
            missing_count = len(missing_files)
            total_count = len(full_paths)
            logging.warning(f"Still missing {missing_count} image files out of {total_count} total")
            logging.warning(f"This should not happen - all molecules should have images (including placeholders)")
            # Still filter just in case
            valid_indices = [i for i, f in enumerate(full_paths) if os.path.exists(f)]
            full_paths = [full_paths[i] for i in valid_indices]
            labels = labels[valid_indices]

        logging.debug(f"Successfully loaded {len(full_paths)} images with {labels.shape[1]} labels")
        return full_paths, labels

    except Exception as e:
        logging.error(f"Error in load_filenames_and_labels: {e}")
        raise


def generate_labels_file(args):
    """
    Generate labels file
    Args:
        args: Arguments object
    """
    processed_file_path = os.path.join(f'data/{args.dataset}/processed/{args.dataset}_processed.csv')
    dataset_path = os.path.join(f'data/{args.dataset}/processed')
    labels_file_path = os.path.join(dataset_path, f'{args.dataset}_label.csv')

    try:
        # Check required files
        if not os.path.exists(processed_file_path):
            raise FileNotFoundError(f"Processed file not found: {processed_file_path}")

        # Read data
        df = pd.read_csv(processed_file_path)

        # Generate filenames
        image_filenames = [f"{i + 1}.png" for i in range(len(df))]

        # Extract label columns
        label_cols = [col for col in df.columns if col not in ['smiles', 'index']]

        # Create labels dataframe
        labels_df = pd.DataFrame({
            "filename": image_filenames,
            **{col: df[col].values for col in label_cols}
        })

        # Save labels file
        os.makedirs(os.path.dirname(labels_file_path), exist_ok=True)
        labels_df.to_csv(labels_file_path, index=False)
        logging.debug(f"Successfully generated labels file: {labels_file_path}")

    except Exception as e:
        logging.error(f"Error generating labels file: {e}")
        raise


def process_dataset(args):
    """
    Process dataset
    Args:
        args: Arguments object
    """
    raw_dataset = os.path.join("data", args.dataset, f'{args.dataset}.csv')
    save_dir = os.path.join("data", args.dataset, "processed")
    save_dataset = os.path.join(save_dir, f'{args.dataset}_processed.csv')

    try:
        # Check input file
        if not os.path.exists(raw_dataset):
            raise FileNotFoundError(f"Raw dataset not found: {raw_dataset}")

        # Read data
        df = pd.read_csv(raw_dataset)

        # Add index
        df.insert(0, "index", range(1, len(df) + 1))

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        # Save processed dataset
        df.to_csv(save_dataset, index=False)
        logging.debug(f"Successfully processed dataset: {save_dataset}")

    except Exception as e:
        logging.error(f"Error processing dataset: {e}")
        raise
