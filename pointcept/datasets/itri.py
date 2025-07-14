"""
NPY Point Cloud dataset for ITRI data - Multi-Domain Support

Author: Modified from Semantic KITTI template
"""

import os
import numpy as np
import glob
import json

from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class ItriDataset(DefaultDataset):
    def __init__(self, 
                 ignore_index=-1,
                 data_root="/home/bryan/pointcloud_data/airport_q2/",
                 domain_names=None,
                 domain_balance=True,
                 **kwargs):
        """
        Args:
            data_root: Single path or list of paths to domain data folders
            domain_names: List of domain names (if None, uses folder names)
            domain_balance: Whether to balance samples across domains (only for multi-domain)
        """
        self.learning_map = self.get_learning_map(ignore_index)
        self.learning_map_inv = self.get_learning_map_inv(ignore_index)
        # Handle both single domain and multi-domain cases
        if isinstance(data_root, str):
            # Single domain case
            self.data_roots = [data_root]
            self.is_multi_domain = False
        else:
            # Multi-domain case
            self.data_roots = data_root
            self.is_multi_domain = True
        
        # Set domain names
        if domain_names is None:
            self.domain_names = [os.path.basename(root.rstrip('/')) for root in self.data_roots]
        else:
            self.domain_names = domain_names
            
        self.domain_balance = domain_balance and self.is_multi_domain
        
        # Validate inputs for multi-domain
        if self.is_multi_domain:
            assert len(self.data_roots) == len(self.domain_names), \
                "Number of data_roots must match number of domain_names"
        
        # Create split files for each domain
        for data_root in self.data_roots:
            self._create_split_files(data_root)
        
        # Use first domain as primary data_root for compatibility
        super().__init__(data_root=self.data_roots[0], **kwargs)

    def _create_split_files(self, data_root):
        """Create split JSON files if they don't exist for a specific domain"""
        if os.path.exists(os.path.join(data_root, "train.json")):
            return
            
        # Get scan NPY files (exclude poses file)
        all_files = glob.glob(os.path.join(data_root, "*.npy"))
        scan_files = []
        
        for f in all_files:
            filename = os.path.basename(f)
            # Skip poses file and only include numeric scan files
            if "_poses.npy" in filename:
                continue
            try:
                # Test if filename (without .npy) is numeric
                int(filename.split('.')[0])
                scan_files.append(filename)
            except ValueError:
                # Skip non-numeric files
                continue
        
        # Sort by scan number
        scan_files.sort(key=lambda x: int(x.split('.')[0]))
        
        # Create splits (70/15/15)
        n = len(scan_files)
        np.random.seed(42)  # Keep consistent across domains
        idx = np.random.permutation(n)
        
        splits = {
            "train": [scan_files[i] for i in idx[:int(n*0.7)]],
            "val": [scan_files[i] for i in idx[int(n*0.7):int(n*0.85)]],
            "test": [scan_files[i] for i in idx[int(n*0.85):]]
        }
        
        # Save split files
        for split_name, files in splits.items():
            with open(os.path.join(data_root, f"{split_name}.json"), 'w') as f:
                json.dump(files, f)
        
        domain_name = os.path.basename(data_root.rstrip('/'))
        print(f"Created splits for {domain_name} from {len(scan_files)} scans: {[(k, len(v)) for k, v in splits.items()]}")

    def get_data_list(self):
        """Get data list - supports both single and multi-domain"""
        if not self.is_multi_domain:
            # Single domain case - use original logic
            if isinstance(self.split, str):
                split_file = os.path.join(self.data_roots[0], f"{self.split}.json")
            else:
                raise NotImplementedError("Multiple splits not supported")
                
            if os.path.exists(split_file):
                with open(split_file) as f:
                    filenames = json.load(f)
                return [os.path.join(self.data_roots[0], fname) for fname in filenames]
            else:
                return []
        
        # Multi-domain case
        all_data = []
        
        for domain_idx, data_root in enumerate(self.data_roots):
            if isinstance(self.split, str):
                split_file = os.path.join(data_root, f"{self.split}.json")
            else:
                raise NotImplementedError("Multiple splits not supported")
                
            if os.path.exists(split_file):
                with open(split_file) as f:
                    filenames = json.load(f)
                
                # Add domain information to each file path
                for fname in filenames:
                    all_data.append({
                        'path': os.path.join(data_root, fname),
                        'domain_idx': domain_idx,
                        'domain_name': self.domain_names[domain_idx],
                        'data_root': data_root
                    })
        
        # Balance domains if requested
        if self.domain_balance and len(self.data_roots) > 1:
            all_data = self._balance_domains(all_data)
        
        print(f"Total samples across {len(self.data_roots)} domains: {len(all_data)}")
        for i, domain_name in enumerate(self.domain_names):
            domain_count = sum(1 for item in all_data if item['domain_idx'] == i)
            print(f"  {domain_name}: {domain_count} samples")
        
        return all_data

    def _balance_domains(self, all_data):
        """Balance the number of samples across domains"""
        # Group by domain
        domain_data = {}
        for item in all_data:
            domain_idx = item['domain_idx']
            if domain_idx not in domain_data:
                domain_data[domain_idx] = []
            domain_data[domain_idx].append(item)
        
        # Find maximum domain size
        max_size = max(len(samples) for samples in domain_data.values())
        
        # Upsample smaller domains by repeating samples
        balanced_data = []
        for domain_idx, samples in domain_data.items():
            # Repeat samples to match max_size
            repeats = max_size // len(samples)
            remainder = max_size % len(samples)
            
            balanced_samples = samples * repeats
            if remainder > 0:
                balanced_samples.extend(samples[:remainder])
            
            balanced_data.extend(balanced_samples)
        
        # Shuffle to mix domains
        np.random.shuffle(balanced_data)
        return balanced_data

    def get_data(self, idx):
        """Load NPY file directly and return expected format"""
        if not self.is_multi_domain:
            # Single domain case - use original logic
            data_path = self.data_list[idx % len(self.data_list)]
            name = self.get_data_name(idx)
            split = self.get_split_name(idx)
            data_root = self.data_roots[0]
            domain_idx = 0
            domain_name = self.domain_names[0]
        else:
            # Multi-domain case
            data_info = self.data_list[idx % len(self.data_list)]
            data_path = data_info['path']
            domain_idx = data_info['domain_idx']
            domain_name = data_info['domain_name']
            data_root = data_info['data_root']
            name = self.get_data_name(idx)
            split = self.get_split_name(idx)
        
        # Load scan data
        scan_data = np.load(data_path)
        
        # Extract components
        coord = scan_data['xyz'].astype(np.float32)
        strength = scan_data['intensity'].reshape(-1, 1).astype(np.float32)
        
        # Map labels
        orig_labels = scan_data['label']['type']
        segment = np.vectorize(self.learning_map.__getitem__)(orig_labels).astype(np.int32)
        
        # Load pose if available
        scan_num = int(os.path.splitext(os.path.basename(data_path))[0])
        pose_file = os.path.join(data_root, f"{os.path.basename(data_root)}_poses.npy")
        pose = None
        if os.path.exists(pose_file):
            poses = np.load(pose_file)
            if scan_num < len(poses):
                pose_data = poses[scan_num]
                # Extract rotation matrix and translation vector
                R = pose_data['R'].astype(np.float32)  # (3, 3)
                t = pose_data['t'].astype(np.float32)  # (3,)
                # Create 4x4 transformation matrix
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = R
                pose[:3, 3] = t
        
        # Build data dict with expected components
        data_dict = {
            "coord": coord,
            "strength": strength,
            "segment": segment.reshape(-1),
            "instance": np.ones(coord.shape[0], dtype=np.int32) * -1,
            "name": name,
            "split": split
        }
        
        # Add domain info for multi-domain case
        if self.is_multi_domain:
            data_dict["domain_idx"] = domain_idx
            data_dict["domain_name"] = domain_name
        
        if pose is not None:
            data_dict["pose"] = pose
        
        return data_dict

    def get_data_name(self, idx):
        """Return scan name - with domain prefix for multi-domain"""
        if not self.is_multi_domain:
            # Single domain case
            file_path = self.data_list[idx % len(self.data_list)]
            return os.path.splitext(os.path.basename(file_path))[0]
        else:
            # Multi-domain case
            data_info = self.data_list[idx % len(self.data_list)]
            file_path = data_info['path']
            domain_name = data_info['domain_name']
            scan_name = os.path.splitext(os.path.basename(file_path))[0]
            return f"{domain_name}_{scan_name}"

    def get_split_name(self, idx):
        """Return current split name"""
        return self.split
    
    @staticmethod
    def get_learning_map(ignore_index):
        """
        Original type -> training class mapping
        """
        learning_map = {
            -1: 0,  # unlabeled -> class 0
            1: 0,    # 'none' -> class 0
            2: 1,    # 'solid' -> class 1  
            3: 2,    # 'broken' -> class 2
            4: 3,    # 'solid solid' -> class 3
            5: 4,    # 'solid broken' -> class 4
            6: 5,    # 'broken solid' -> class 5
            7: 6,    # 'broken broken' -> class 6
            8: 7,    # 'botts dots' -> class 7
            9: 8,    # 'grass' -> class 8
            10: 9,   # 'curb' -> class 9
            11: 10,  # 'custom' -> class 10
            12: 11,  # 'edge' -> class 11
        }
        return learning_map

    @staticmethod
    def get_learning_map_inv(ignore_index):
        """
        Training class -> original type mapping (for visualization)
        """
        learning_map_inv = {
            # ignore_index: ignore_index,  # ignore -> ignore
            0: -1,    # class 0 -> 'none' (type 1)
            1: 2,    # class 1 -> 'solid' (type 2)
            2: 3,    # class 2 -> 'broken' (type 3)
            3: 4,    # class 3 -> 'solid solid' (type 4)
            4: 5,    # class 4 -> 'solid broken' (type 5)
            5: 6,    # class 5 -> 'broken solid' (type 6)
            6: 7,    # class 6 -> 'broken broken' (type 7)
            7: 8,    # class 7 -> 'botts dots' (type 8)
            8: 9,    # class 8 -> 'grass' (type 9)
            9: 10,   # class 9 -> 'curb' (type 10)
            10: 11,  # class 10 -> 'custom' (type 11)
            11: 12,  # class 11 -> 'edge' (type 12)
        }
        return learning_map_inv

    def convert_predictions_to_original(self, predictions):
        """
        Convert model predictions back to original label types for visualization
        
        Args:
            predictions: numpy array or torch tensor with training class predictions (0-11)
            
        Returns:
            numpy array with original type labels (1-12)
        """
        # Convert to numpy if torch tensor
        if hasattr(predictions, 'cpu'):  # torch tensor
            predictions = predictions.cpu().numpy()
        
        # Apply inverse learning map
        original_types = np.vectorize(self.learning_map_inv.__getitem__)(predictions)
        
        return original_types.astype(np.int32)

    @classmethod
    def get_class_names(cls):
        """Get human-readable class names for training classes 0-11"""
        return [
            'none',         # class 0
            'solid',        # class 1
            'broken',       # class 2
            'solid_solid',  # class 3
            'solid_broken', # class 4
            'broken_solid', # class 5
            'broken_broken',# class 6
            'botts_dots',   # class 7
            'grass',        # class 8
            'curb',         # class 9
            'custom',       # class 10
            'edge'          # class 11
        ]

    @classmethod
    def get_original_type_names(cls):
        """Get human-readable names for original types 1-12"""
        return {
            1: 'none',
            2: 'solid',
            3: 'broken',
            4: 'solid_solid',
            5: 'solid_broken',
            6: 'broken_solid',
            7: 'broken_broken',
            8: 'botts_dots',
            9: 'grass',
            10: 'curb',
            11: 'custom',
            12: 'edge'
        }