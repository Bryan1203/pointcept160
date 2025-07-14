#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, '/home/itri464058/pointcept160')

from pointcept.datasets.itri import ItriDataset

def test_single_domain():
    print("Testing single domain dataset...")
    
    # Test single domain
    dataset = ItriDataset(
        split='train',
        data_root='/home/itri464058/pointcloud_data/hsinchu_q2/',
        transform=[],  # No transforms for basic test
        test_mode=False,
        ignore_index=-1
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    if len(dataset) > 0:
        print("Testing data loading...")
        try:
            sample = dataset[0]
            print(f"First sample keys: {sample.keys()}")
            if 'coord' in sample:
                print(f"Coord shape: {sample['coord'].shape}")
            if 'strength' in sample:
                print(f"Strength shape: {sample['strength'].shape}")
            if 'segment' in sample:
                print(f"Segment shape: {sample['segment'].shape}")
                print(f"Unique labels: {set(sample['segment'].flatten())}")
            
            # Test a few more samples
            for i in range(min(3, len(dataset))):
                sample = dataset[i]
                print(f"Sample {i}: coord {sample['coord'].shape}, segment {sample['segment'].shape}")
                
        except Exception as e:
            print(f"Error loading sample: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Dataset is empty!")

def test_multi_domain():
    print("\nTesting multi-domain dataset...")
    
    # Test multi-domain
    data_roots = [
        '/home/itri464058/pointcloud_data/hsinchu_q2/',
        '/home/itri464058/pointcloud_data/airport_q2/'
    ]
    domain_names = ['hsinchu_q2', 'airport_q2']
    
    try:
        dataset = ItriDataset(
            split='train',
            data_root=data_roots,
            domain_names=domain_names,
            domain_balance=False,
            transform=[],  # No transforms for basic test
            test_mode=False,
            ignore_index=-1
        )
        
        print(f"Multi-domain dataset length: {len(dataset)}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"First sample keys: {sample.keys()}")
            if 'coord' in sample:
                print(f"Coord shape: {sample['coord'].shape}")
            if 'domain' in sample:
                print(f"Domain: {sample['domain']}")
        
    except Exception as e:
        print(f"Error with multi-domain dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting dataset tests...")
    test_single_domain()
    test_multi_domain()
    print("Tests completed.")
