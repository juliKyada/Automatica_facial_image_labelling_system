# Automatic Facial Image Labelling System  

##  Project Overview  
The **Automatic Facial Image Labelling System** is a machine learning pipeline designed to automatically generate labels (age, gender, emotion, ethnicity, etc.) for unlabeled facial images.  
This system leverages **semi-supervised learning** where a small labeled dataset guides the model to label large amounts of unlabeled data, improving efficiency and scalability.  

---

## Features  
- ✅ Supports **labeled + unlabeled dataset integration**  
- ✅ Generates **pseudo-labels** for unlabeled facial images  
- ✅ Trains models for **Age, Gender, Emotion, and Ethnicity classification**  
- ✅ Uses **semi-supervised learning** (Pseudo-Labeling, FixMatch style)  
- ✅ Includes **Streamlit GUI** for interactive processing  
- ✅ Outputs structured labeled datasets ready for further training  

---

##  System Architecture  
1. **Input**  
   - Labeled Dataset (UTKFace, FER2019, or custom labeled dataset)  
   - Unlabeled Dataset (raw face images without labels)  

2. **Processing**  
   - Preprocessing & resizing (64×64 or 128×128)  
   - Predictions using pre-trained models  
   - Filtering high-confidence pseudo-labels  

3. **Output**  
   - Combined dataset (labeled + pseudo-labeled)  
   - Final dataset ready for supervised training  

---

##  Dataset  
- **Labeled Dataset:** UTKFace, FER2019, or custom-labeled images  
- **Unlabeled Dataset:** Any collection of facial images (no labels required)  

> ⚠️ Note: Ensure images are stored in `.jpg` / `.png` format inside `Dataset/labeled/` and `Dataset/unlabeled/`.  

---

##  Installation  

Clone the repository and install requirements:  

```bash
git clone https://github.com/yourusername/automatic-facial-labelling.git
cd automatic-facial-labelling
pip install -r requirements.txt
