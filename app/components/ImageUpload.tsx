'use client'

import { useState, useRef } from 'react'
import axios from 'axios'
import styles from './ImageUpload.module.css'

interface ImageUploadProps {
  onPredict: (result: any) => void
  loading: boolean
  setLoading: (loading: boolean) => void
}

export default function ImageUpload({ onPredict, loading, setLoading }: ImageUploadProps) {
  const [image, setImage] = useState<string | null>(null)
  const [imageFile, setImageFile] = useState<File | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setImageFile(file)
      const reader = new FileReader()
      reader.onloadend = () => {
        setImage(reader.result as string)
      }
      reader.readAsDataURL(file)
    }
  }

  const handlePredict = async () => {
    if (!imageFile) return

    setLoading(true)
    try {
      // Convert image to base64
      const reader = new FileReader()
      reader.onloadend = async () => {
        const base64Image = reader.result as string
        
        try {
          // Use absolute URL for production, relative for development
          const apiUrl = process.env.NODE_ENV === 'production' 
            ? '/api/predict' 
            : '/api/predict'
          
          const response = await axios.post(apiUrl, {
            image: base64Image
          }, {
            headers: {
              'Content-Type': 'application/json'
            }
          })
          
          onPredict(response.data)
        } catch (error: any) {
          console.error('Prediction error:', error)
          alert(`Error: ${error.response?.data?.error || error.message || 'Failed to predict'}`)
        } finally {
          setLoading(false)
        }
      }
      reader.readAsDataURL(imageFile)
    } catch (error) {
      console.error('Error:', error)
      setLoading(false)
    }
  }

  return (
    <div className={styles.uploadContainer}>
      <div 
        className="upload-area"
        onClick={() => fileInputRef.current?.click()}
      >
        {image ? (
          <div className={styles.imagePreview}>
            <img src={image} alt="Preview" className={styles.previewImage} />
            <p>Click to change image</p>
          </div>
        ) : (
          <div>
            <p style={{ fontSize: '1.1rem', fontWeight: '500', marginBottom: '0.5rem' }}>Click to upload an image</p>
            <p style={{ fontSize: '0.9rem', color: '#666666', marginTop: '0.5rem' }}>
              Supported formats: JPG, PNG, BMP
            </p>
          </div>
        )}
      </div>
      
      <input
        ref={fileInputRef}
        type="file"
        accept="image/jpeg,image/jpg,image/png,image/bmp"
        onChange={handleFileChange}
        style={{ display: 'none' }}
      />
      
      {image && (
        <button
          className="button"
          onClick={handlePredict}
          disabled={loading}
          style={{ marginTop: '1.5rem', width: '100%', padding: '1rem' }}
        >
          {loading ? 'Processing...' : 'Analyze Image'}
        </button>
      )}
    </div>
  )
}

