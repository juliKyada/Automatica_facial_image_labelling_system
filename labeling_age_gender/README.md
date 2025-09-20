#  Automatic Facial Image Labelling System

A professional GUI application for automatic age and gender prediction from facial images using deep learning models trained with pseudo-labelling techniques.

## Features

- **Modern GUI Interface**: Clean, professional design with intuitive controls
- **Single Image Analysis**: Upload and analyze individual images with detailed results
- **Batch Processing**: Process multiple images simultaneously with progress tracking
- **Real-time Predictions**: Instant age and gender predictions using pre-trained models
- **Results Export**: Export batch results to CSV or Excel format
- **Model Status Monitoring**: Real-time display of model loading and status
- **Professional Styling**: Modern color scheme and responsive design

##  Installation

### Prerequisites
- Python 3.7 or higher
- Windows 10/11 (tested on Windows 10)

### Setup
1. Clone or download this repository
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

##  Project Structure

```
labeling_age_gender/
├── facial_labelling_gui.py    # Main GUI application
├── age_gender_pseudolabel.h5  # Pre-trained age/gender model
├── requirements.txt            # Python dependencies
├── README.md                  # This file
└── Automatic_image_labelling.ipynb  # Original training notebook
```

##  Usage

### Starting the Application
```bash
python facial_labelling_gui.py
```

### Single Image Analysis
1. Click **" Upload Image"** to select an image file
2. Click **" Predict"** to analyze the image
3. View detailed results including:
   - Predicted age
   - Predicted gender
   - Confidence scores
   - Raw model outputs

### Batch Processing
1. Click **" Batch Process"** to select a folder of images
2. Monitor progress with the progress bar
3. View results in the "Batch Results" tab
4. Export results using **" Export Results"**

### Supported Image Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)

## 🔧 Technical Details

### Model Architecture
- **Age Prediction**: Regression model with MSE loss
- **Gender Prediction**: Binary classification with binary crossentropy loss
- **Input Size**: 48x48 pixels (RGB)
- **Preprocessing**: Resize to 48x48, normalize to [0,1]

### Pseudo-labelling Approach
The system uses a small amount of manually labelled data combined with predictions from pre-trained models to create a larger training dataset, improving model performance through semi-supervised learning.

### Performance Metrics
- **Age Prediction**: Mean Absolute Error (MAE)
- **Gender Prediction**: Binary accuracy
- **Model Validation**: 20% test split with cross-validation

##  Results Interpretation

### Age Prediction
- Output: Integer age in years (0-100)
- Confidence: Based on model's regression output variance

### Gender Prediction
- Output: Male/Female classification
- Confidence: Probability score (0.0-1.0)
- Threshold: 0.5 (above = Male, below = Female)

##  Troubleshooting

### Common Issues

1. **Models not loading**
   - Ensure `age_gender_pseudolabel.h5` is in the same directory
   - Check TensorFlow installation and version compatibility

2. **Memory errors during batch processing**
   - Reduce batch size or process fewer images at once
   - Close other applications to free up memory

3. **Image loading errors**
   - Verify image file integrity
   - Check supported file formats
   - Ensure sufficient disk space

### Performance Tips
- Use SSD storage for faster image loading
- Close unnecessary applications during batch processing
- Process images in smaller batches for large datasets

##  Future Enhancements

- [ ] Emotion recognition integration
- [ ] Ethnicity prediction
- [ ] Real-time webcam analysis
- [ ] Advanced visualization tools
- [ ] Model fine-tuning interface
- [ ] Cloud deployment options

## 📝 License

This project is developed for research and educational purposes. Please ensure compliance with relevant data privacy and usage regulations.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📞 Support

For technical support or questions, please refer to the original training notebook or create an issue in the repository.

---

**Note**: This application requires the pre-trained model file (`age_gender_pseudolabel.h5`) to function. Ensure the model file is present in the project directory before running the GUI.
