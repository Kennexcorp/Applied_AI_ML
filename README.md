# Applied AI/ML

A collection of machine learning projects demonstrating various techniques in artificial intelligence and machine learning, with a focus on computer vision using Convolutional Neural Networks (CNNs).

## What I've Learned

Through these projects, I gained hands-on experience with:

- **CNN Architecture**: Understanding convolution layers, pooling, feature maps, and how CNNs extract hierarchical features from images
- **Image Classification**: From basic digit recognition to complex multi-class wildlife identification and audio classification via spectrograms
- **Transfer Learning**: Leveraging pre-trained models (ResNet, MobileNet, VGGFace) to adapt large-scale trained networks to specific tasks
- **Data Augmentation**: Using Keras preprocessing layers (RandomFlip, RandomRotation, RandomZoom) to artificially expand datasets and improve model generalization
- **Face Detection & Recognition**: Implementing Viola-Jones algorithm with OpenCV and modern MTCNN for face detection, plus transfer learning approaches for facial recognition
- **Object Detection**: Working with Mask R-CNN for instance segmentation, bounding boxes, and background replacement using ONNX models
- **Audio Processing**: Converting audio signals to spectrogram images using librosa for CNN-based sound classification
- **Data Preprocessing**: Image loading, resizing, normalization, augmentation techniques, and audio-to-image transformations
- **Model Training**: Gradient descent, loss functions (categorical cross-entropy), optimization (Adam), and monitoring training/validation accuracy
- **Pre-trained Models**: Using ImageNet-trained CNNs for custom prediction tasks and domain-specific models (VGGFace) for specialized applications
- **Keras API**: Building sequential models, adding layers, compiling models, and fitting to data with integrated preprocessing
- **TensorFlow Ecosystem**: Working with preprocessing utilities, applications models, and ONNX runtime for deployment
- **Computer Vision Pipeline**: From basic image processing to advanced object detection and facial recognition systems
- **Model Evaluation**: Using confusion matrices and accuracy metrics to assess classification performance

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

### 5. Audio Classification (`CNN - Audio Classificaton.ipynb`)
- Converted audio files (WAV) into visual spectrograms using `librosa`
- Transformed an audio classification problem into an image classification task
- Trained a CNN on the generated spectrogram images to classify different sounds (background, chainsaw, engine, storm)

### 6. Face Detection (`OpenCv Viola-Jonnes - Face Detection.ipynb`)
- Implemented Viola-Jones algorithm for face detection using OpenCV
- Utilized pre-trained Haar Cascade classifiers (`CascadeClassifier`)
- Detected faces in images and visualized results with bounding boxes

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
