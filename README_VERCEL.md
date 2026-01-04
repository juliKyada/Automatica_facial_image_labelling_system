# Automatic Facial Image Labelling System - Vercel Deployment

A modern web application for automatic facial image analysis using deep learning models, deployed on Vercel.

## 🚀 Features

- **Age Prediction**: Regression-based age estimation (0-100 years)
- **Gender Classification**: Binary classification (Male/Female) with confidence scores
- **Ethnicity Recognition**: Multi-class ethnicity classification (optional)
- **Emotion Detection**: Facial emotion recognition (optional)
- **Modern UI**: Built with Next.js and React
- **Cloud Deployment**: Ready for Vercel deployment

## 📋 Prerequisites

- Node.js 18+ and npm
- Vercel account (free tier works for testing)
- Model files (`.h5` files)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd labeling_age_gender
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Ensure model files are in place**:
   - `labeling_age_gender/age_gender_pseudolabel.h5` (required)
   - `labeling_age_gender/Ethnicity_lebelling.h5` (optional)
   - `labeling_age_gender/emotion_model.h5` (optional)

## 🏃 Running Locally

1. **Start the development server**:
   ```bash
   npm run dev
   ```

2. **Open your browser**:
   Navigate to [http://localhost:3000](http://localhost:3000)

## 📦 Deployment to Vercel

See [VERCEL_DEPLOYMENT.md](./VERCEL_DEPLOYMENT.md) for detailed deployment instructions.

### Quick Deploy

1. **Install Vercel CLI**:
   ```bash
   npm i -g vercel
   ```

2. **Deploy**:
   ```bash
   vercel
   ```

3. **Deploy to production**:
   ```bash
   vercel --prod
   ```

## ⚠️ Important Notes

### Model File Size Limitations

Vercel serverless functions have size limits:
- **Hobby Plan**: 50MB per function
- **Pro Plan**: 100MB per function

If your model files exceed these limits, consider:
1. Storing models in external cloud storage (S3, GCS)
2. Using model quantization to reduce size
3. Upgrading to Vercel Pro plan

### Function Timeout

- **Hobby Plan**: 10 seconds
- **Pro Plan**: 60 seconds

Model loading and prediction may take time. Consider caching models in memory.

## 📁 Project Structure

```
labeling_age_gender/
├── app/                    # Next.js app directory
│   ├── components/         # React components
│   │   ├── ImageUpload.tsx
│   │   ├── ResultsDisplay.tsx
│   │   └── Navigation.tsx
│   ├── layout.tsx          # Root layout
│   ├── page.tsx            # Home page
│   └── globals.css          # Global styles
├── api/                     # Python API routes
│   ├── predict.py          # Prediction endpoint
│   ├── utils.py            # Utility functions
│   └── requirements.txt     # Python dependencies
├── labeling_age_gender/     # Model files
│   ├── age_gender_pseudolabel.h5
│   ├── Ethnicity_lebelling.h5
│   └── emotion_model.h5
├── package.json
├── vercel.json              # Vercel configuration
├── next.config.js
└── tsconfig.json
```

## 🔧 Configuration

### API Endpoint

The prediction API is available at `/api/predict` and accepts:
- **Method**: POST
- **Content-Type**: application/json
- **Body**: `{ "image": "base64_encoded_image" }`

### Model Paths

Update model paths in `api/predict.py` if needed:
```python
MODEL_BASE_PATH = os.path.join(os.path.dirname(__file__), '..', 'labeling_age_gender')
```

## 🧪 Testing

### Test the API

```bash
# Convert image to base64
base64_image=$(base64 -i test_image.jpg)

# Send prediction request
curl -X POST http://localhost:3000/api/predict \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"data:image/jpeg;base64,$base64_image\"}"
```

## 📊 API Response Format

```json
{
  "success": true,
  "age": 25,
  "gender": "Female",
  "gender_confidence": 0.95,
  "ethnicity": {
    "label": "Class_0",
    "confidence": 0.87
  },
  "emotion": {
    "label": "Happy",
    "confidence": 0.92
  }
}
```

## 🐛 Troubleshooting

### Models not loading
- Check model file paths
- Verify model files are in the correct directory
- Check Vercel function logs

### Function timeout
- Model loading takes time on first request
- Subsequent requests are faster (models cached)
- Consider increasing timeout in `vercel.json`

### Large model files
- Use external storage (S3, GCS)
- Implement model downloading on first request
- Cache models in memory

## 📝 License

This project is for research and educational purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

---

**Note**: This is a converted version of the original Streamlit application, now optimized for Vercel deployment with Next.js and React.

