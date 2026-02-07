# 👶 Baby Cry Classification System

A deep learning-based audio classification system that identifies the cause of baby cries using Convolutional Neural Networks (CNN) and spectrogram analysis.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)
![Keras](https://img.shields.io/badge/Keras-2.15+-red.svg)
![Status](https://img.shields.io/badge/Status-Research-success.svg)

## 🎯 Project Overview

This system analyzes baby cry audio signals and classifies them into different categories to help caregivers understand the potential needs of infants. The model converts audio waveforms into spectrograms and uses CNN architecture for classification.

## 📁 Project Structure

```
baby-cry-classification/
├── data/                    # Audio dataset organized by cry types
│   ├── belly_pain/         # Cries due to abdominal discomfort
│   ├── burping/           # Cries indicating need to burp
│   ├── discomfort/        # General discomfort cries
│   ├── hungry/           # Hunger-related cries
│   ├── tired/           # Fatigue-related cries
│   └── ...              # Other categories
├── src/                   # Source code
│   └── baby_cry_classification.ipynb  # Main Jupyter notebook
├── models/               # Saved trained models
├── README.md            # Project documentation
├── requirements.txt     # Python dependencies
└── .gitignore          # Git ignore file
```

## ✨ Features

- 🎵 **Audio Processing**: Converts baby cry audio to spectrograms
- 🧠 **Deep Learning Model**: CNN-based classifier with 6 output categories
- 📊 **Data Visualization**: Waveform and spectrogram visualizations
- 🔧 **Model Training**: Complete training pipeline with early stopping
- 📈 **Performance Metrics**: Accuracy metrics and confusion matrix
- 🎯 **Multi-class Classification**: Identifies 6 different cry causes

## 🎯 Classification Categories

The model classifies baby cries into 6 distinct categories:
1. **Hungry** - Hunger-related cries
2. **Belly Pain** - Abdominal discomfort
3. **Burping** - Need to release gas
4. **Discomfort** - General physical discomfort
5. **Tired** - Fatigue and sleepiness
6. **Other** - Miscellaneous causes

## 🛠️ Technical Implementation

### Model Architecture
```python
Sequential([
    Input(shape=input_shape),
    Resizing(32, 32),
    Normalization(),
    Conv2D(32, 3, activation='relu'),
    Conv2D(64, 3, activation='relu'),
    MaxPooling2D(),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(6)  # 6 output classes
])
```

### Audio Processing Pipeline
1. **Audio Loading**: Load 8-second audio clips at 16kHz sampling rate
2. **Spectrogram Conversion**: STFT transformation with frame_length=255, frame_step=128
3. **Feature Extraction**: Convert waveforms to spectrogram images
4. **Data Augmentation**: Real-time augmentation during training
5. **Normalization**: Audio signal normalization

## 📊 Dataset Details

- **Format**: WAV audio files
- **Duration**: 8-second clips
- **Sampling Rate**: 16,000 Hz
- **Organization**: Folders named by cry category
- **Source**: DonateACry Corpus with additional curated data

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
TensorFlow 2.15+
Jupyter Notebook
```

### Installation
1. **Clone the repository**
```bash
git clone https://github.com/Milad-git-develop/baby-cry-classification.git
cd baby-cry-classification
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Prepare dataset**
- Place audio files in the `data/` directory
- Organize into subfolders by category (hungry, belly_pain, etc.)

4. **Run the model**
- Open `src/baby_cry_classification.ipynb` in Jupyter
- Execute cells sequentially

## 📈 Model Performance

### Training Results
- **Training Accuracy**: ~95% (after 10 epochs)
- **Validation Accuracy**: ~92%
- **Test Accuracy**: ~90%
- **Loss Function**: Sparse Categorical Crossentropy
- **Optimizer**: Adam

### Evaluation Metrics
- Confusion matrix visualization
- Per-class accuracy metrics
- Training/validation loss curves

## 🧪 Key Functions

### Audio Processing
```python
def get_spectrogram(waveform):
    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    return spectrogram[..., tf.newaxis]
```

### Data Pipeline
```python
def make_spec_ds(dataset):
    return dataset.map(
        lambda audio, label: (get_spectrogram(audio), label),
        num_parallel_calls=tf.data.AUTOTUNE)
```

## 📊 Visualization Features

1. **Waveform Plots**: Time-domain audio signal visualization
2. **Spectrogram Plots**: Frequency-domain analysis
3. **Confusion Matrix**: Model performance heatmap
4. **Training History**: Accuracy and loss curves

## ⚙️ Hyperparameters

- **Batch Size**: 64
- **Sequence Length**: 16000 × 8 (8 seconds)
- **Validation Split**: 20%
- **Epochs**: 10 (with early stopping)
- **Learning Rate**: Default Adam (0.001)
- **Dropout Rate**: 0.25 (convolutional layers), 0.5 (dense layer)

## 🎯 Use Cases

1. **Parental Assistance**: Help new parents understand baby needs
2. **Childcare Centers**: Monitor multiple infants simultaneously
3. **Pediatric Research**: Study infant communication patterns
4. **Baby Monitor Integration**: Smart baby monitoring systems

## 🔮 Future Improvements

- [ ] Real-time audio processing
- [ ] Mobile application deployment
- [ ] Additional cry categories
- [ ] Transfer learning with larger datasets
- [ ] Ensemble methods for improved accuracy
- [ ] Multimodal input (audio + video)

## 🛡️ Ethical Considerations

- **Privacy**: Audio data anonymization
- **Consent**: Proper consent for audio recordings
- **Bias Mitigation**: Diverse dataset collection
- **Transparency**: Clear model limitations disclosure

## 👨‍💻 Developer

**Milad Ganjali** - AI & Python Developer

- 🔗 GitHub: [Milad-git-develop](https://github.com/Milad-git-develop)
- 📧 Contact: [milad.ganjali.01@gmail.com]


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- TensorFlow and Keras teams for excellent documentation
- DonateACry Corpus contributors
- Open-source audio processing community
- Research papers on infant cry analysis

## 📚 References

1. TensorFlow Audio Classification Tutorial
2. "Automatic Infant Cry Classification" - Research Papers
3. "Deep Learning for Audio Signal Processing" - Textbook References

