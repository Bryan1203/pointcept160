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
                 concat_scans=1,
                 **kwargs):
        """
        Args:
            data_root: Single path or list of paths to domain data folders
            domain_names: List of domain names (if None, uses folder names)
            domain_balance: Whether to balance samples across domains (only for multi-domain)
            concat_scans: Number of consecutive scans to concatenate (default: 1)
        """
        self.learning_map = self.get_learning_map(ignore_index)
        self.learning_map_inv = self.get_learning_map_inv(ignore_index)
        self.concat_scans = concat_scans  # Number of scans to concatenate
        
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
        
        # Create splits (70/15/15) - SEQUENTIAL to preserve temporal continuity for concatenation
        n = len(scan_files)
        
        # Use sequential splits instead of random to maintain temporal order
        # This ensures consecutive scans stay together for concatenation
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)
        
        splits = {
            "train": scan_files[:train_end],
            "val": scan_files[train_end:val_end], 
            "test": scan_files[val_end:]
        }
        
        # Save split files
        for split_name, files in splits.items():
            with open(os.path.join(data_root, f"{split_name}.json"), 'w') as f:
                json.dump(files, f)
        
        domain_name = os.path.basename(data_root.rstrip('/'))
        print(f"Created splits for {domain_name} from {len(scan_files)} scans: {[(k, len(v)) for k, v in splits.items()]}")

    def get_data_list(self):
        """Get data list - supports both single and multi-domain with scan concatenation"""
        if not self.is_multi_domain:
            # Single domain case - use original logic
            if isinstance(self.split, str):
                split_file = os.path.join(self.data_roots[0], f"{self.split}.json")
            else:
                raise NotImplementedError("Multiple splits not supported")
                
            if os.path.exists(split_file):
                with open(split_file) as f:
                    filenames = json.load(f)
                
                # Sort filenames by scan number for sequential concatenation
                filenames.sort(key=lambda x: int(x.split('.')[0]))
                
                # Group scans for concatenation - non-overlapping groups
                if self.concat_scans > 1:
                    grouped_data = []
                    
                    # Create non-overlapping groups of scans
                    for i in range(0, len(filenames), self.concat_scans):
                        # Only create group if we have enough scans remaining
                        if i + self.concat_scans <= len(filenames):
                            scan_group = []
                            for j in range(self.concat_scans):
                                scan_group.append(os.path.join(self.data_roots[0], filenames[i + j]))
                            grouped_data.append(scan_group)
                    
                    return grouped_data
                else:
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
                
                # Sort filenames by scan number for sequential concatenation
                filenames.sort(key=lambda x: int(x.split('.')[0]))
                
                # Group scans for concatenation - non-overlapping groups
                if self.concat_scans > 1:
                    
                    # Create non-overlapping groups of scans
                    for i in range(0, len(filenames), self.concat_scans):
                        # Only create group if we have enough scans remaining
                        if i + self.concat_scans <= len(filenames):
                            scan_group = []
                            for j in range(self.concat_scans):
                                scan_group.append(os.path.join(data_root, filenames[i + j]))
                            
                            all_data.append({
                                'paths': scan_group,  # List of scan paths
                                'domain_idx': domain_idx,
                                'domain_name': self.domain_names[domain_idx],
                                'data_root': data_root
                            })
                else:
                    # Single scan case
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
        if self.concat_scans > 1:
            print(f"Each sample contains {self.concat_scans} concatenated scans")
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
        """Load NPY file(s) and concatenate if needed, return expected format"""
        if not self.is_multi_domain:
            # Single domain case
            if self.concat_scans > 1:
                # Multiple scans case
                scan_paths = self.data_list[idx % len(self.data_list)]
                name = self.get_data_name(idx)
                split = self.get_split_name(idx)
                data_root = self.data_roots[0]
                domain_idx = 0
                domain_name = self.domain_names[0]
                
                # Load and concatenate multiple scans
                return self._load_and_concatenate_scans(scan_paths, name, split, data_root, domain_idx, domain_name)
            else:
                # Single scan case - use original logic
                data_path = self.data_list[idx % len(self.data_list)]
                name = self.get_data_name(idx)
                split = self.get_split_name(idx)
                data_root = self.data_roots[0]
                domain_idx = 0
                domain_name = self.domain_names[0]
                
                return self._load_single_scan(data_path, name, split, data_root, domain_idx, domain_name)
        else:
            # Multi-domain case
            data_info = self.data_list[idx % len(self.data_list)]
            domain_idx = data_info['domain_idx']
            domain_name = data_info['domain_name']
            data_root = data_info['data_root']
            name = self.get_data_name(idx)
            split = self.get_split_name(idx)
            
            if self.concat_scans > 1:
                # Multiple scans case
                scan_paths = data_info['paths']
                return self._load_and_concatenate_scans(scan_paths, name, split, data_root, domain_idx, domain_name)
            else:
                # Single scan case
                data_path = data_info['path']
                return self._load_single_scan(data_path, name, split, data_root, domain_idx, domain_name)

    def _load_single_scan(self, data_path, name, split, data_root, domain_idx, domain_name):
        """Load a single scan"""
        # Load scan data
        scan_data = np.load(data_path)
        
        # Extract components
        coord = scan_data['xyz'].astype(np.float32)
        strength = scan_data['intensity'].reshape(-1, 1).astype(np.float32)
        
        # Map labels
        orig_labels = scan_data['label']['type']
        segment = np.vectorize(self.learning_map.__getitem__)(orig_labels).astype(np.int32)
        
        # Load pose if available and apply transformation
        scan_num = int(os.path.splitext(os.path.basename(data_path))[0])
        # pose_file = os.path.join(data_root, f"{os.path.basename(data_root.rstrip('/'))}_poses.npy")
        pose_file = os.path.join(data_root, "scan_poses.npy")
        pose = None
        if os.path.exists(pose_file):
            poses = np.load(pose_file)
            
            # Find pose by matching index field to scan number
            pose_idx = None
            for i, pose_entry in enumerate(poses):
                if pose_entry['index'] == scan_num:
                    pose_idx = i
                    break
            
            if pose_idx is not None:
                pose_data = poses[pose_idx]
                # Extract rotation matrix and translation vector
                R = pose_data['R'].astype(np.float32)  # (3, 3)
                t = pose_data['t'].astype(np.float32)  # (3,)
                
                # Store original center for debugging
                orig_center = np.mean(coord, axis=0)
                
                # Apply pose transformation to coordinates (transform from global to local coordinates)
                # Using inverse transformation: local = R.T @ (global - t)
                coord = np.dot(R.T, (coord - t).T).T
                
                # Check transformed center
                new_center = np.mean(coord, axis=0)
                print(f"  Single scan {scan_num}: orig center [{orig_center[0]:.2f}, {orig_center[1]:.2f}, {orig_center[2]:.2f}] -> local center [{new_center[0]:.2f}, {new_center[1]:.2f}, {new_center[2]:.2f}]")
                
                # Create 4x4 transformation matrix for metadata
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = R
                pose[:3, 3] = t
            else:
                print(f"  Warning: Scan {scan_num} not found in poses")
        else:
            print(f"  Warning: Pose file not found for single scan: {pose_file}")
        
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

    def _load_and_concatenate_scans(self, scan_paths, name, split, data_root, domain_idx, domain_name):
        """Load and concatenate multiple scans with pose-based transformation relative to first scan"""
        all_coords = []
        all_strengths = []
        all_segments = []
        all_instances = []
        
        # Load poses
        # pose_file = os.path.join(data_root, f"{os.path.basename(data_root.rstrip('/'))}_poses.npy")
        pose_file = os.path.join(data_root, "scan_poses.npy")
        poses = None
        reference_R = None
        reference_t = None
        
        if os.path.exists(pose_file):
            poses = np.load(pose_file)
            
            # Get reference transformation from the FIRST scan in the batch
            first_scan_num = int(os.path.splitext(os.path.basename(scan_paths[0]))[0])
            
            # Find pose by matching index field to scan number
            reference_pose_idx = None
            for i, pose_entry in enumerate(poses):
                if pose_entry['index'] == first_scan_num:
                    reference_pose_idx = i
                    break
            
            if reference_pose_idx is not None:
                first_pose_data = poses[reference_pose_idx]
                reference_R = first_pose_data['R'].astype(np.float32)  # (3, 3)
                reference_t = first_pose_data['t'].astype(np.float32)  # (3,)
                print(f"  Using scan {first_scan_num} as reference origin for concatenation")
                print(f"  Reference pose - R shape: {reference_R.shape}, t: [{reference_t[0]:.2f}, {reference_t[1]:.2f}, {reference_t[2]:.2f}]")
            else:
                print(f"  Warning: First scan {first_scan_num} not found in poses")
                reference_R = None
                reference_t = None
        else:
            print(f"  Warning: Pose file not found: {pose_file}")
        
        for scan_idx, scan_path in enumerate(scan_paths):
            # Load scan data
            scan_data = np.load(scan_path)
            
            # Extract components
            coord = scan_data['xyz'].astype(np.float32)
            strength = scan_data['intensity'].reshape(-1, 1).astype(np.float32)
            
            # Map labels
            orig_labels = scan_data['label']['type']
            segment = np.vectorize(self.learning_map.__getitem__)(orig_labels).astype(np.int32)
            
            # Transform coordinates to the reference scan's coordinate frame
            if poses is not None and reference_R is not None and reference_t is not None:
                # Store original center for debugging
                orig_center = np.mean(coord, axis=0)
                
                # Transform ALL scans (including the reference scan) to the reference scan's origin
                # This matches the working code's transform_to_origin function approach
                # Formula: transformed = R_ref.T @ (original - t_ref)
                coord = np.dot(reference_R.T, (coord - reference_t).T).T
                
                # Check transformed center
                new_center = np.mean(coord, axis=0)
                
                scan_num = int(os.path.splitext(os.path.basename(scan_path))[0])
                if scan_idx == 0:
                    print(f"    First scan {scan_num}: orig center [{orig_center[0]:.2f}, {orig_center[1]:.2f}, {orig_center[2]:.2f}] -> ref center [{new_center[0]:.2f}, {new_center[1]:.2f}, {new_center[2]:.2f}]")
                    print(f"    Note: First scan should be close to origin after transformation")
                else:
                    print(f"    Scan {scan_num}: orig center [{orig_center[0]:.2f}, {orig_center[1]:.2f}, {orig_center[2]:.2f}] -> ref center [{new_center[0]:.2f}, {new_center[1]:.2f}, {new_center[2]:.2f}]")
            else:
                scan_num = int(os.path.splitext(os.path.basename(scan_path))[0])
                print(f"    Warning: No pose transformation applied for scan {scan_num}")
            
            instance = np.ones(coord.shape[0], dtype=np.int32) * (scan_idx + 1)  # Start from 1
            
            all_coords.append(coord)
            all_strengths.append(strength)
            all_segments.append(segment)
            all_instances.append(instance)
        
        # Concatenate all scans
        final_coord = np.concatenate(all_coords, axis=0)
        final_strength = np.concatenate(all_strengths, axis=0)
        final_segment = np.concatenate(all_segments, axis=0)
        final_instance = np.concatenate(all_instances, axis=0)
        
        # Get reference scan number for metadata
        reference_scan_num = int(os.path.splitext(os.path.basename(scan_paths[0]))[0])
        
        # Build data dict
        data_dict = {
            "coord": final_coord,
            "strength": final_strength,
            "segment": final_segment.reshape(-1),
            "instance": final_instance,
            "name": name,
            "split": split,
            "concat_scans": self.concat_scans,  # Add info about concatenation
            "scan_paths": scan_paths,  # Keep track of source scans
            "reference_scan": reference_scan_num,  # Which scan is used as reference origin
            "scan_boundaries": [0] + [sum(len(coord) for coord in all_coords[:i+1]) for i in range(len(all_coords))]  # Point boundaries for each scan
        }
        
        # Add domain info for multi-domain case
        if self.is_multi_domain:
            data_dict["domain_idx"] = domain_idx
            data_dict["domain_name"] = domain_name
        
        return data_dict

    def get_data_name(self, idx):
        """Return scan name - with domain prefix for multi-domain and concatenation info"""
        if not self.is_multi_domain:
            # Single domain case
            if self.concat_scans > 1:
                scan_paths = self.data_list[idx % len(self.data_list)]
                scan_nums = [os.path.splitext(os.path.basename(path))[0] for path in scan_paths]
                return f"concat_{'-'.join(scan_nums)}"
            else:
                file_path = self.data_list[idx % len(self.data_list)]
                return os.path.splitext(os.path.basename(file_path))[0]
        else:
            # Multi-domain case
            data_info = self.data_list[idx % len(self.data_list)]
            domain_name = data_info['domain_name']
            
            if self.concat_scans > 1:
                scan_paths = data_info['paths']
                scan_nums = [os.path.splitext(os.path.basename(path))[0] for path in scan_paths]
                return f"{domain_name}_concat_{'-'.join(scan_nums)}"
            else:
                file_path = data_info['path']
                scan_name = os.path.splitext(os.path.basename(file_path))[0]
                return f"{domain_name}_{scan_name}"

    def get_split_name(self, idx):
        """Return current split name"""
        return self.split
    
    @staticmethod
    def get_learning_map(ignore_index):
        """
        Original type -> training class mapping with robust default handling
        """
        from collections import defaultdict
        
        # Use defaultdict to map any unknown labels to class 0 for robustness
        learning_map = defaultdict(lambda: 0)
        
        # Explicit mappings
        learning_map.update({
            -1: 0,  # unlabeled -> class 0
            0: 0,   # handle 0 labels -> class 0
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
        })
        
        # Map range 101-158 to class 0 (except specific overrides below)
        # for i in range(101, 158 + 1):
        #     learning_map[i] = 0
            
        # Specific overrides for certain types
        learning_map.update({
            116: 12,
            118: 12,
            142: 13,
            143: 13,
            144: 13,
            145: 13,
            146: 13,
            147: 13,
            148: 13,
            149: 13,
            109: 14,
            121: 15,
            156: 15,
            137: 16,
            138: 16,
            139: 16,
        })
        
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
            12: 116, # class 12 -> type 116 (first occurrence)
            13: 142, # class 13 -> type 142 (first occurrence)
            14: 109, # class 14 -> type 109
            15: 121, # class 15 -> type 121 (first occurrence)
            16: 137, # class 16 -> type 137 (first occurrence)
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
    
    def split_concatenated_predictions(self, predictions, data_dict):
        """
        Split concatenated predictions back to individual scans
        
        Args:
            predictions: Model predictions for concatenated scans (N_total,)
            data_dict: Data dictionary containing scan boundaries and metadata
            
        Returns:
            List of dictionaries, one for each scan with individual predictions
        """
        if 'scan_boundaries' not in data_dict or 'scan_paths' not in data_dict:
            # Single scan case, return as-is
            return [{
                'predictions': predictions,
                'scan_path': data_dict.get('name', 'unknown'),
                'scan_num': data_dict.get('name', 'unknown')
            }]
        
        scan_boundaries = data_dict['scan_boundaries']
        scan_paths = data_dict['scan_paths']
        
        individual_predictions = []
        
        for i in range(len(scan_paths)):
            start_idx = scan_boundaries[i]
            end_idx = scan_boundaries[i + 1]
            
            # Extract predictions for this scan
            scan_predictions = predictions[start_idx:end_idx]
            
            # Get scan number from path
            scan_num = int(os.path.splitext(os.path.basename(scan_paths[i]))[0])
            
            individual_predictions.append({
                'predictions': scan_predictions,
                'scan_path': scan_paths[i],
                'scan_num': scan_num,
                'domain_name': data_dict.get('domain_name', 'unknown'),
                'start_idx': start_idx,
                'end_idx': end_idx
            })
        
        return individual_predictions

    def save_individual_predictions(self, predictions, data_dict, output_dir, save_format='npy'):
        """
        Save predictions for each individual scan separately
        
        Args:
            predictions: Model predictions for concatenated scans
            data_dict: Data dictionary containing scan metadata
            output_dir: Directory to save individual prediction files
            save_format: Format to save ('npy', 'txt', or 'both')
        """
        individual_preds = self.split_concatenated_predictions(predictions, data_dict)
        
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = []
        for scan_info in individual_preds:
            scan_num = scan_info['scan_num']
            scan_predictions = scan_info['predictions']
            domain_name = scan_info.get('domain_name', 'unknown')
            
            # Create filename
            if self.is_multi_domain:
                base_filename = f"{domain_name}_{scan_num}"
            else:
                base_filename = f"{scan_num}"
            
            # Save in requested format(s)
            if save_format in ['npy', 'both']:
                npy_path = os.path.join(output_dir, f"{base_filename}_pred.npy")
                np.save(npy_path, scan_predictions)
                saved_files.append(npy_path)
                
            if save_format in ['txt', 'both']:
                txt_path = os.path.join(output_dir, f"{base_filename}_pred.txt")
                np.savetxt(txt_path, scan_predictions, fmt='%d')
                saved_files.append(txt_path)
                
        return saved_files