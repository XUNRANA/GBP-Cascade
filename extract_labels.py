import os
import glob
import json

def extract_labels(data_dir='dataset'):
    patients = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d.isdigit()]
    labels = {'malignant': [], 'benign': [], 'non_neoplastic': [], 'unknown': []}

    for p in patients:
        pathology_file = glob.glob(os.path.join(data_dir, p, '*_pathology.txt'))
        if not pathology_file:
            continue
        
        pathology_file = pathology_file[0]
        
        try:
            with open(pathology_file, 'r', encoding='gb18030', errors='ignore') as f:
                text = f.read()
                
                # Simple keyword-based extraction based on the task description
                if '癌' in text or '恶性' in text:
                    labels['malignant'].append(p)
                elif '腺瘤' in text:
                    labels['benign'].append(p)
                elif '息肉' in text or '炎' in text or '结石' in text or '囊肿' in text or '胆固醇' in text:
                    labels['non_neoplastic'].append(p)
                else:
                    labels['unknown'].append(p)
        except Exception as e:
            pass
            
    print(f"Total patients analyzed: {len(patients)}")
    print(f"Malignant (Class 2): {len(labels['malignant'])}")
    print(f"Benign Adenoma (Class 1): {len(labels['benign'])}")
    print(f"Non-neoplastic (Class 0): {len(labels['non_neoplastic'])}")
    print(f"Unknown/Unclassified: {len(labels['unknown'])}")
    
    # Save the labels for the dataset loader
    with open('dataset_labels.json', 'w') as f:
        json.dump(labels, f)
        
    return labels

if __name__ == '__main__':
    extract_labels()
    
    try:
        import torch
        import torchvision
        print(f"PyTorch version: {torch.__version__}, GPU available: {torch.cuda.is_available()}")
    except ImportError:
        print("PyTorch is not installed. Please install it using 'pip install torch torchvision'.")
