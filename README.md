## 🧠 Project Overview & Role (Machine Learning / Computer Vision Engineer)

I designed, trained, and evaluated an end-to-end deep learning pipeline in PyTorch to classify 7 types of skin lesions using the **HAM10000** dermatoscopic image dataset:

* **Data Pipeline & Augmentation:** Built a custom PyTorch `Dataset` and `DataLoader` pipeline with stratified train/validation/test splits, image normalization, and data augmentation (random resized crops, flips, rotations, and color jitter) to improve model generalization.
* **Class Imbalance Handling:** Computed balanced class weights (`compute_class_weight`) and integrated them into `CrossEntropyLoss` to counteract heavy dataset skew across rare lesion categories.
* **Two-Phase Transfer Learning (`ResNet50`):** Implemented a pretrained ResNet50 architecture—first training the custom 7-class classifier head with frozen backbone weights, then fine-tuning the full network (`lr=1e-4`), reaching **80.87% training accuracy**.
* **Model Optimization (Pruning):** Applied L1 unstructured weight pruning (`30%` sparsity) to the fully connected layer to reduce model parameter redundancy.
