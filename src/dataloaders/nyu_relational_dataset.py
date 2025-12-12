"""
nyu_relational_dataset.py
NYU-Depth-v2 Dataset with Relational Annotations
WorDepth + RelationalDepthLoss를 위한 Dataset 클래스
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import platform


class NYURelationalDataset(Dataset):
    """
    NYU-Depth-v2 with relational annotations (masks + relations)
    
    Structure:
        nyu-depth-v2/train/scene_name/rgb_XXXXX.png
        nyu-depth-v2/train/scene_name/depth_XXXXX.png
        nyu-processed/train/scene_name/rgb_XXXXX_masks.npy
        nyu-processed/train/scene_name/rgb_XXXXX_relations.json
    """
    
    def __init__(self, 
                 filenames_file=None,  # Optional: if None, auto-scan directories
                 data_path=None,
                 gt_path=None,
                 relations_base_path=None,
                 is_train=True,
                 input_height=480,
                 input_width=640,
                 max_depth=10.0,
                 do_random_rotate=False,
                 degree=2.5,
                 use_relational_loss=True):
        """
        Args:
            filenames_file: path to filenames list (optional, if None auto-scans directories)
            data_path: path to RGB images (nyu-depth-v2/train)
            gt_path: path to depth GT (same as data_path for NYU)
            relations_base_path: path to processed relations (nyu-processed/train)
            is_train: training mode
            use_relational_loss: whether to load masks and relations
        """
        self.is_train = is_train
        # Convert relative paths to absolute paths for Windows compatibility
        self.data_path = os.path.abspath(data_path) if data_path and not os.path.isabs(data_path) else data_path
        self.gt_path = os.path.abspath(gt_path) if gt_path and not os.path.isabs(gt_path) else gt_path
        self.relations_base_path = relations_base_path
        if relations_base_path and not os.path.isabs(relations_base_path):
            self.relations_base_path = os.path.abspath(relations_base_path)
        self.use_relational_loss = use_relational_loss
        
        self.input_height = input_height
        self.input_width = input_width
        self.max_depth = max_depth
        
        self.do_random_rotate = do_random_rotate
        self.degree = degree
        
        # Auto-scan directories if filenames_file not provided
        if filenames_file is None or filenames_file == '':
            self.filenames = self._auto_scan_rgb_depth_pairs()
        else:
            # Read filenames from file
            with open(filenames_file, 'r') as f:
                self.filenames = f.readlines()
        
        # Transforms
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def _auto_scan_rgb_depth_pairs(self):
        """
        Automatically scan directories and match RGB-Depth pairs by filename.
        Returns list of pairs in format: "scene_name/rgb_XXXXX.png scene_name/depth_XXXXX.png"
        """
        pairs = []
        
        if not os.path.exists(self.data_path):
            raise ValueError(f"Data path does not exist: {self.data_path}")
        
        # Scan all scene directories
        scene_dirs = [d for d in os.listdir(self.data_path) 
                     if os.path.isdir(os.path.join(self.data_path, d))]
        
        print(f"Auto-scanning {len(scene_dirs)} scenes in {self.data_path}...")
        
        for scene_name in sorted(scene_dirs):
            scene_dir = os.path.join(self.data_path, scene_name)
            
            # Find all RGB files (PNG only - actual data format)
            rgb_files = [f for f in os.listdir(scene_dir) 
                        if f.startswith('rgb_') and f.endswith('.png')]
            
            # Find all depth files (PNG only - actual data format)
            depth_files = [f for f in os.listdir(scene_dir) 
                          if f.startswith('depth_') and f.endswith('.png')]
            
            # Match RGB and Depth by number
            for rgb_file in sorted(rgb_files):
                # Extract number from rgb_XXXXX.png
                import re
                rgb_match = re.search(r'rgb_(\d+)', rgb_file)
                if not rgb_match:
                    continue
                rgb_num = rgb_match.group(1)
                
                # Find matching depth file
                depth_file = None
                for d_file in depth_files:
                    if f'depth_{rgb_num}' in d_file or f'sync_depth_{rgb_num}' in d_file:
                        depth_file = d_file
                        break
                
                if depth_file:
                    # Format: "scene_name/rgb_XXXXX.png scene_name/depth_XXXXX.png"
                    pair = f"{scene_name}/{rgb_file} {scene_name}/{depth_file}"
                    pairs.append(pair)
        
        print(f"Found {len(pairs)} RGB-Depth pairs")
        return pairs
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        sample_path = self.filenames[idx].strip()
        
        # Parse filename (format: "/scene_name/rgb_XXXXX.jpg /scene_name/sync_depth_XXXXX.png focal")
        parts = sample_path.split()
        rgb_file = parts[0]  # e.g., "/kitchen_0028b/rgb_00045.jpg"
        depth_file = parts[1] if len(parts) > 1 else None  # e.g., "/kitchen_0028b/sync_depth_00045.png"
        
        # Remove leading slash if present (filenames_file uses absolute-style paths)
        rgb_file = rgb_file.lstrip('/')
        if depth_file:
            depth_file = depth_file.lstrip('/')
        
        # Normalize path separators for Windows compatibility
        rgb_file = rgb_file.replace('/', os.sep)
        if depth_file:
            depth_file = depth_file.replace('/', os.sep)
        
        # Normalize: ensure .png extension (actual data format is PNG)
        # If filenames_file has .jpg, convert to .png
        if rgb_file.endswith('.jpg'):
            rgb_file = rgb_file.replace('.jpg', '.png')
        elif not rgb_file.endswith('.png'):
            rgb_file = rgb_file + '.png'
        
        if depth_file:
            # Try depth_ first (actual format), then sync_depth_ (filenames_file format)
            if 'sync_depth_' in depth_file:
                depth_file_alt = depth_file.replace('sync_depth_', 'depth_')
                # Use depth_ format (actual data)
                depth_file = depth_file_alt
            if depth_file.endswith('.jpg'):
                depth_file = depth_file.replace('.jpg', '.png')
            elif not depth_file.endswith('.png'):
                depth_file = depth_file + '.png'
        
        # Full paths
        image_path = os.path.join(self.data_path, rgb_file)
        if depth_file:
            depth_path = os.path.join(self.gt_path, depth_file)
        else:
            # Fallback: try to construct depth path from rgb path
            depth_path = os.path.join(self.gt_path, rgb_file.replace('rgb_', 'depth_'))
        
        # Normalize paths (resolve ./, .., etc.)
        image_path = os.path.normpath(image_path)
        depth_path = os.path.normpath(depth_path)
        
        # Try different depth file names if the original doesn't exist
        if not os.path.exists(depth_path):
            # Try depth_XXXXX.png instead of sync_depth_XXXXX.png
            if 'sync_depth_' in depth_path:
                depth_path_alt = depth_path.replace('sync_depth_', 'depth_')
                if os.path.exists(depth_path_alt):
                    depth_path = depth_path_alt
        
        # Check if files exist, if not, try to find closest match or raise clear error
        if not os.path.exists(image_path):
            # Try to find any rgb file in the same directory as fallback
            scene_dir = os.path.dirname(image_path)
            if os.path.exists(scene_dir):
                rgb_files = [f for f in os.listdir(scene_dir) if f.startswith('rgb_') and f.endswith('.png')]
                if rgb_files:
                    # Use first available file as fallback (for debugging)
                    print(f"Warning: {os.path.basename(image_path)} not found in {scene_dir}, using {rgb_files[0]} instead")
                    image_path = os.path.join(scene_dir, rgb_files[0])
                    # Update rgb_file for text feature path
                    rgb_file = os.path.join(os.path.basename(scene_dir), rgb_files[0]).replace(os.sep, '/')
                else:
                    raise FileNotFoundError(f"RGB file not found: {image_path} (and no rgb_*.png files in {scene_dir})")
            else:
                raise FileNotFoundError(f"Scene directory not found: {scene_dir}")
        
        if not os.path.exists(depth_path):
            scene_dir = os.path.dirname(depth_path)
            if os.path.exists(scene_dir):
                depth_files = [f for f in os.listdir(scene_dir) if f.startswith('depth_') and f.endswith('.png')]
                if depth_files:
                    # Match by number if possible
                    rgb_basename = os.path.basename(image_path).replace('rgb_', '').replace('.png', '').replace('.jpg', '')
                    matching_depth = [f for f in depth_files if rgb_basename in f]
                    if matching_depth:
                        depth_path = os.path.join(scene_dir, matching_depth[0])
                    else:
                        print(f"Warning: {os.path.basename(depth_path)} not found, using {depth_files[0]} instead")
                        depth_path = os.path.join(scene_dir, depth_files[0])
                else:
                    raise FileNotFoundError(f"Depth file not found: {depth_path} (and no depth_*.png files in {scene_dir})")
            else:
                raise FileNotFoundError(f"Scene directory not found: {scene_dir}")
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Load depth (NYU-matched: 16-bit PNG where 65535 = 10m)
        depth_gt = Image.open(depth_path)
        depth_gt = np.array(depth_gt, dtype=np.float32)
        depth_gt = depth_gt / 6553.5  # 16-bit (0-65535) -> 0-10m
        depth_gt = np.expand_dims(depth_gt, axis=0)  # (1, H, W)
        
        # Resize if needed
        if image.size != (self.input_width, self.input_height):
            image = image.resize((self.input_width, self.input_height), Image.BILINEAR)
            depth_gt_pil = Image.fromarray(depth_gt[0])
            depth_gt_pil = depth_gt_pil.resize((self.input_width, self.input_height), Image.NEAREST)
            depth_gt = np.array(depth_gt_pil)[np.newaxis, :, :]
        
        # Random rotation (training only)
        if self.is_train and self.do_random_rotate:
            angle = (np.random.random() - 0.5) * 2 * self.degree
            image = image.rotate(angle, resample=Image.BILINEAR)
            depth_gt_pil = Image.fromarray(depth_gt[0])
            depth_gt_pil = depth_gt_pil.rotate(angle, resample=Image.NEAREST)
            depth_gt = np.array(depth_gt_pil)[np.newaxis, :, :]
        
        # To tensor
        image = self.to_tensor(image)
        image = self.normalize(image)
        depth_gt = torch.from_numpy(depth_gt).float()
        
        # Valid depth mask
        has_valid_depth = torch.any(depth_gt > 0.1)
        
        # For text feature loading in train.py, use the format expected by existing code
        # train.py expects: sample_path in format "/scene/rgb_XXXXX.jpg /scene/depth_XXXXX.png"
        # We keep sample_path as original, but use rgb_file (actual loaded file) for relations
        
        # Base sample
        sample = {
            'image': image,
            'depth': depth_gt,
            'sample_path': sample_path,  # Original path from filenames_file (for text feature)
            'has_valid_depth': has_valid_depth
        }
        
        # Load relational annotations if needed
        if self.use_relational_loss and self.relations_base_path:
            masks, relations = self._load_relational_annotations(rgb_file)
            sample['masks'] = masks
            sample['relations'] = relations
        
        return sample
    
    def _load_relational_annotations(self, rgb_file):
        """
        Load masks and relations for a given RGB file
        
        Args:
            rgb_file: e.g., "kitchen_0028b/rgb_00045.png" (OS-specific separator)
        
        Returns:
            masks: torch.Tensor (N_obj, H, W) or None
            relations: list[dict] or None
        """
        # Normalize to forward slash for path construction
        rgb_file_unix = rgb_file.replace(os.sep, '/')
        
        # Construct paths
        scene_name = os.path.dirname(rgb_file_unix)
        rgb_basename = os.path.basename(rgb_file_unix).replace('.jpg', '').replace('.png', '')
        
        mask_path = os.path.join(
            self.relations_base_path,
            scene_name,
            f"{rgb_basename}_masks.npy"
        )
        rel_path = os.path.join(
            self.relations_base_path,
            scene_name,
            f"{rgb_basename}_relations.json"
        )
        
        # Load masks
        try:
            masks_np = np.load(mask_path)  # (N_obj, H, W)
            
            # Resize masks if needed
            if masks_np.shape[1:] != (self.input_height, self.input_width):
                import torch.nn.functional as F
                masks_tensor = torch.from_numpy(masks_np).float().unsqueeze(1)  # (N, 1, H, W)
                masks_tensor = F.interpolate(
                    masks_tensor,
                    size=(self.input_height, self.input_width),
                    mode='nearest'
                )
                masks_tensor = masks_tensor.squeeze(1)  # (N, H, W)
            else:
                masks_tensor = torch.from_numpy(masks_np).float()
            
            # Apply binary threshold for cleaner masks
            masks_tensor = (masks_tensor > 0.5).float()
            
        except Exception as e:
            # If masks not found, return None
            masks_tensor = None
        
        # Load relations
        try:
            with open(rel_path, 'r') as f:
                relations = json.load(f)
            
            # Filter out unsupported relations ('above' is not supported by RelationalDepthLoss)
            if relations is not None:
                relations = [rel for rel in relations if rel.get('relation', '').lower() != 'above']
        except Exception as e:
            # If relations not found, return None
            relations = None
        
        return masks_tensor, relations


def collate_fn_with_relations(batch):
    """
    Custom collate function for batching with variable-size masks
    
    Args:
        batch: list of samples from dataset
    
    Returns:
        batched_sample: dict with batched tensors
    """
    # Regular fields - can be stacked normally
    images = torch.stack([item['image'] for item in batch])
    depths = torch.stack([item['depth'] for item in batch])
    sample_paths = [item['sample_path'] for item in batch]
    has_valid_depth = torch.stack([item['has_valid_depth'] for item in batch])
    
    batched_sample = {
        'image': images,
        'depth': depths,
        'sample_path': sample_paths,
        'has_valid_depth': has_valid_depth
    }
    
    # Relational annotations - need special handling
    if 'masks' in batch[0]:
        masks_list = []
        relations_list = []
        
        for item in batch:
            masks = item.get('masks')
            relations = item.get('relations')
            
            # Handle None cases
            if masks is None or relations is None:
                # Create empty placeholders
                masks_list.append(torch.zeros((0, item['image'].shape[1], item['image'].shape[2])))
                relations_list.append([])
            else:
                masks_list.append(masks)
                relations_list.append(relations)
        
        # Keep as list (simpler, works with variable number of objects per image)
        batched_sample['masks'] = masks_list
        batched_sample['relations'] = relations_list
    
    return batched_sample


def create_nyu_relational_dataloader(args, mode='train'):
    """
    Create DataLoader for NYU-Depth-v2 with relational annotations
    
    Args:
        args: training arguments
        mode: 'train' or 'online_eval'
    
    Returns:
        DataLoader
    """
    is_train = (mode == 'train')
    
    if is_train:
        filenames_file = getattr(args, 'filenames_file', None)
        data_path = args.data_path
        gt_path = args.gt_path
        relations_path = getattr(args, 'relations_dir_train', None)
    else:
        filenames_file = getattr(args, 'filenames_file_eval', None)
        data_path = args.data_path_eval
        gt_path = args.gt_path_eval
        relations_path = getattr(args, 'relations_dir_eval', None)
    
    # Check if relational loss is enabled
    use_relational_loss = getattr(args, 'use_relational_loss', False)
    
    if use_relational_loss and relations_path is None:
        print("Warning: use_relational_loss=True but relations directory not specified!")
        print("         Relations will not be loaded.")
        use_relational_loss = False
    
    # Create dataset
    dataset = NYURelationalDataset(
        filenames_file=filenames_file,
        data_path=data_path,
        gt_path=gt_path,
        relations_base_path=relations_path if use_relational_loss else None,
        is_train=is_train,
        input_height=args.input_height,
        input_width=args.input_width,
        max_depth=args.max_depth,
        do_random_rotate=getattr(args, 'do_random_rotate', False),
        degree=getattr(args, 'degree', 2.5),
        use_relational_loss=use_relational_loss
    )
    
    # Create dataloader
    # Windows multiprocessing issue: use num_workers=0 on Windows to avoid yaml import errors
    num_workers = 0 if platform.system() == 'Windows' else getattr(args, 'num_threads', 4)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size if is_train else 1,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn_with_relations if use_relational_loss else None
    )
    
    # Wrap in object with .data attribute for compatibility with NewDataLoader
    class DataLoaderWrapper:
        def __init__(self, dataloader):
            self.data = dataloader
    
    return DataLoaderWrapper(dataloader)
