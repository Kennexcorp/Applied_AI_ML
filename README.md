# Applied AI/ML

A collection of machine learning projects demonstrating various techniques in artificial intelligence and machine learning, with a focus on computer vision using Convolutional Neural Networks (CNNs).

## What I've Learned

Through these projects, I gained hands-on experience with:

- **CNN Architecture**: Understanding convolution layers, pooling, feature maps, and how CNNs extract hierarchical features from images
- **Image Classification**: From basic digit recognition to complex multi-class wildlife identification
- **Transfer Learning**: Leveraging pre-trained models like ResNet and MobileNet to adapt large-scale trained networks to specific tasks
- **Data Preprocessing**: Image loading, resizing, normalization, and data augmentation techniques for CNN training
- **Model Training**: Gradient descent, loss functions (categorical cross-entropy), optimization (Adam), and monitoring training/validation accuracy
- **Pre-trained CNNs**: Using models trained on ImageNet (1000 classes) for custom image prediction tasks
- **Keras API**: Building sequential models, adding layers, compiling models, and fitting to data
- **TensorFlow Ecology**: Working with image preprocessing utilities and application models in TensorFlow/Keras

## Projects

### 1. MNIST Digit Classification (`CNN - Minst Image Classification.ipynb`)
- Learned basic CNN building blocks: Conv2D, MaxPooling2D, Flatten, Dense layers
- Implemented sequential model architecture for grayscale image classification
- Experienced training dynamics and plotting accuracy curves

### 2. Arctic Wildlife Image Classification (`CNN - Arctic Wildlife Image Classification.ipynb`)
- Built deeper CNN with 5 pairs of conv/pool layers for RGB images
- Practiced loading custom datasets from directory structures
- Understood feature extraction at multiple resolutions (224→111→54→26→12 pixels)

### 3. Pretrained CNNs (`Pretrained CNNs.ipynb`)
- Discovered Keras Applications module with dozens of pre-trained models
- Learned proper preprocessing for different model architectures (MobileNet, ResNet)
- Predicted classes for novel images using ImageNet-trained models

### 4. Transfer Learning for Arctic Wildlife (`CNN Transfer Learning- Arctic Wildlife Image Classification.ipynb`)
- Implemented feature extraction technique: used ResNet50V2 bottleneck layers to generate features
- Trained custom classifier on top of extracted features
- Combined pre-trained feature extractors with task-specific classification layers

## Additional Projects
- **Titanic Survival Prediction**: Machine learning model for predicting passenger survival
- **Sentiment Analysis**: Natural language processing model for text classification

## Datasets
- **MNIST**: 28×28 grayscale handwritten digits (10 classes)
- **Arctic Wildlife**: 224×224 RGB images of arctic fox, polar bear, walrus (3 classes)
- **Additional**: Various CSV datasets for supervised learning tasks

## Prerequisites
- Python 3.10+
- TensorFlow/Keras
- NumPy, Matplotlib, Seaborn
- Pillow, Jupyter Notebook

## Running the Projects
1. Ensure data directories are properly set up
2. Execute Jupyter notebook cells sequentially
3. Observe training progress and evaluate model performance

## Technologies Used
- **Deep Learning**: TensorFlow/Keras CNNs and pre-trained models
- **Visualization**: Matplotlib plots for training curves
- **Data Handling**: NumPy arrays, Keras image utilities

## Repository
https://github.com/Kennexcorp/Applied_AI_ML
