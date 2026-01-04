# Local Development Setup

## Python Dependencies

For local development, you need to install Python dependencies in the same environment that your Next.js app will use.

### Option 1: Install in System Python (Recommended for Quick Start)

```bash
pip install tensorflow opencv-python-headless numpy Pillow
```

Or if you're using Python 3 specifically:
```bash
pip3 install tensorflow opencv-python-headless numpy Pillow
```

### Option 2: Use Virtual Environment (Recommended for Production)

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   ```

2. **Activate it**:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`

3. **Install dependencies**:
   ```bash
   pip install tensorflow opencv-python-headless numpy Pillow
   ```

4. **Update the route.ts** to use the virtual environment Python:
   - The script will try to auto-detect Python with TensorFlow
   - Or you can manually specify the path in `app/api/predict/route.ts`

### Option 3: Use the Same Environment as Streamlit

If you already have the Streamlit app working, use the same Python environment:

1. **Find which Python your Streamlit uses**:
   ```bash
   python -c "import sys; print(sys.executable)"
   ```

2. **Use that Python path** in the Next.js route, or ensure it's in your PATH

## Testing

After installing dependencies, test if TensorFlow is accessible:

```bash
python -c "import tensorflow; print('TensorFlow version:', tensorflow.__version__)"
```

If this works, your Next.js app should be able to use it too.

## Troubleshooting

### Error: "No module named 'tensorflow'"

- Make sure you installed TensorFlow in the Python environment that `python` or `python3` points to
- Check which Python is being used: `python --version` or `python3 --version`
- If you have multiple Python installations, you may need to use the full path to the Python with TensorFlow

### Error: "Module not found" for other packages

Install all required packages:
```bash
pip install tensorflow opencv-python-headless numpy Pillow pandas
```

### Using a Specific Python Version

If you need to use a specific Python version, you can modify `app/api/predict/route.ts` to use the full path:

```typescript
const pythonCmd = 'C:\\Python39\\python.exe'  // Windows example
// or
const pythonCmd = '/usr/bin/python3'  // Linux/Mac example
```

