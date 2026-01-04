# Vercel Deployment Guide

This guide will help you deploy the Automatic Facial Image Labelling System to Vercel.

## Prerequisites

1. **Vercel Account**: Sign up at [vercel.com](https://vercel.com)
2. **Git Repository**: Your code should be in a Git repository (GitHub, GitLab, or Bitbucket)
3. **Model Files**: Ensure your model files are accessible:
   - `age_gender_pseudolabel.h5` (required)
   - `Ethnicity_lebelling.h5` (optional)
   - `emotion_model.h5` (optional)

## Important Considerations

### Model File Size
- Vercel serverless functions have a **50MB size limit** per function
- TensorFlow model files can be large (often 50-200MB+)
- **Solution Options**:
  1. **Use Vercel Pro Plan**: Higher limits (100MB per function)
  2. **Store models externally**: Upload models to cloud storage (S3, Google Cloud Storage) and download on-demand
  3. **Use TensorFlow.js**: Convert models to TensorFlow.js format and run client-side (limited functionality)
  4. **Optimize models**: Use model quantization or pruning to reduce size

### Recommended Approach
For production, we recommend storing model files in cloud storage and downloading them on first request, then caching them in memory.

## Deployment Steps

### Option 1: Deploy via Vercel CLI

1. **Install Vercel CLI**:
   ```bash
   npm i -g vercel
   ```

2. **Login to Vercel**:
   ```bash
   vercel login
   ```

3. **Install dependencies**:
   ```bash
   npm install
   ```

4. **Deploy**:
   ```bash
   vercel
   ```
   Follow the prompts to link your project.

5. **Deploy to production**:
   ```bash
   vercel --prod
   ```

### Option 2: Deploy via GitHub

1. **Push your code to GitHub**:
   ```bash
   git add .
   git commit -m "Prepare for Vercel deployment"
   git push origin main
   ```

2. **Import to Vercel**:
   - Go to [vercel.com/new](https://vercel.com/new)
   - Import your GitHub repository
   - Vercel will auto-detect Next.js
   - Click "Deploy"

3. **Configure Environment Variables** (if needed):
   - Go to Project Settings → Environment Variables
   - Add any required variables

## Project Structure

```
labeling_age_gender/
├── app/                    # Next.js app directory
│   ├── components/         # React components
│   ├── layout.tsx          # Root layout
│   ├── page.tsx            # Home page
│   └── globals.css          # Global styles
├── api/                     # Python API routes
│   ├── predict.py          # Prediction endpoint
│   ├── utils.py            # Utility functions
│   └── requirements.txt    # Python dependencies
├── labeling_age_gender/     # Model files directory
│   ├── age_gender_pseudolabel.h5
│   ├── Ethnicity_lebelling.h5
│   └── emotion_model.h5
├── package.json            # Node.js dependencies
├── vercel.json             # Vercel configuration
└── next.config.js          # Next.js configuration
```

## Configuration

### vercel.json
The `vercel.json` file configures:
- Python runtime for API routes
- Function timeout (30 seconds)
- Routing rules

### Model Path Configuration
Update `api/predict.py` if your model files are in a different location:
```python
MODEL_BASE_PATH = os.path.join(os.path.dirname(__file__), '..', 'labeling_age_gender')
```

## Handling Large Model Files

### Option A: External Storage (Recommended)

1. **Upload models to cloud storage** (e.g., AWS S3, Google Cloud Storage)

2. **Update `api/predict.py`** to download models:
   ```python
   import boto3  # or google-cloud-storage
   
   def download_model_from_s3(bucket, key, local_path):
       s3 = boto3.client('s3')
       s3.download_file(bucket, key, local_path)
   ```

3. **Cache models in memory** after first download

### Option B: Git LFS (Git Large File Storage)

1. **Install Git LFS**:
   ```bash
   git lfs install
   ```

2. **Track model files**:
   ```bash
   git lfs track "*.h5"
   ```

3. **Add and commit**:
   ```bash
   git add .gitattributes
   git add *.h5
   git commit -m "Add model files with LFS"
   ```

**Note**: Vercel supports Git LFS, but there may still be size limits.

## Testing Locally

1. **Install dependencies**:
   ```bash
   npm install
   ```

2. **Run development server**:
   ```bash
   npm run dev
   ```

3. **Test API endpoint**:
   ```bash
   curl -X POST http://localhost:3000/api/predict \
     -H "Content-Type: application/json" \
     -d '{"image": "base64_encoded_image_data"}'
   ```

## Troubleshooting

### Error: "Function exceeded maximum size"
- **Solution**: Use external storage for model files or upgrade to Vercel Pro

### Error: "Function timeout"
- **Solution**: Increase timeout in `vercel.json` (max 60s on Pro plan)

### Error: "Model file not found"
- **Solution**: Check `MODEL_BASE_PATH` in `api/predict.py` matches your file structure

### Error: "Module not found"
- **Solution**: Ensure all dependencies are in `api/requirements.txt`

## Performance Optimization

1. **Model Caching**: Models are cached in global variables after first load
2. **Image Preprocessing**: Optimized preprocessing functions
3. **Lazy Loading**: Models loaded only when needed

## Cost Considerations

- **Hobby Plan**: Free, but limited (10s timeout, 50MB function size)
- **Pro Plan**: $20/month (60s timeout, 100MB function size, better for production)
- **Enterprise**: Custom pricing for high-scale deployments

## Next Steps

1. Deploy to Vercel
2. Test the deployed application
3. Monitor function logs in Vercel dashboard
4. Optimize model loading if needed
5. Set up custom domain (optional)

## Support

For issues or questions:
- Check Vercel documentation: [vercel.com/docs](https://vercel.com/docs)
- Review function logs in Vercel dashboard
- Check model file paths and sizes

