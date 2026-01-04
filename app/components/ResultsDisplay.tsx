'use client'

import { useState } from 'react'
import styles from './ResultsDisplay.module.css'

interface ResultsDisplayProps {
  results: any
}

export default function ResultsDisplay({ results }: ResultsDisplayProps) {
  const [downloading, setDownloading] = useState(false)

  if (!results || !results.success) {
    return null
  }

  const handleDownload = () => {
    if (!results.labeled_image) return
    
    setDownloading(true)
    try {
      // Convert base64 to blob
      const base64Data = results.labeled_image.split(',')[1]
      const byteCharacters = atob(base64Data)
      const byteNumbers = new Array(byteCharacters.length)
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i)
      }
      const byteArray = new Uint8Array(byteNumbers)
      const blob = new Blob([byteArray], { type: 'image/jpeg' })
      
      // Create download link
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `labeled_image_${new Date().getTime()}.jpg`
      link.style.display = 'none'
      
      // Safely append and remove
      if (document.body) {
        document.body.appendChild(link)
        link.click()
        
        // Use setTimeout to ensure click completes before removal
        setTimeout(() => {
          if (link && link.parentNode) {
            link.parentNode.removeChild(link)
          }
          URL.revokeObjectURL(url)
        }, 100)
      }
    } catch (error) {
      console.error('Download error:', error)
      alert('Failed to download image')
    } finally {
      setDownloading(false)
    }
  }

  return (
    <div className={styles.resultsContainer}>
      <h3 className="sub-header">Analysis Results</h3>
      
      {/* Display labeled image if available */}
      {results.labeled_image && (
        <div className={styles.labeledImageContainer}>
          <div className={styles.imageHeader}>
            <h4>Labeled Image</h4>
            <button
              onClick={handleDownload}
              disabled={downloading}
              className="button"
            >
              {downloading ? 'Downloading...' : 'Download Image'}
            </button>
          </div>
          <img 
            src={results.labeled_image} 
            alt="Labeled prediction" 
            className={styles.labeledImage}
          />
        </div>
      )}
      
      <div className={styles.resultsGrid}>
        <div className="metric-card">
          <h4>Age</h4>
          <p className={styles.metricValue}>{results.age} years</p>
        </div>
        
        <div className="metric-card">
          <h4>Gender</h4>
          <p className={styles.metricValue}>{results.gender}</p>
          <div className={styles.confidenceBar}>
            <div 
              className={styles.confidenceFill}
              style={{ width: `${results.gender_confidence * 100}%` }}
            />
          </div>
          <p className={styles.confidenceText}>
            {(results.gender_confidence * 100).toFixed(1)}% confidence
          </p>
        </div>
      </div>

      {results.ethnicity && (
        <div className="info-box">
          <h4>Ethnicity</h4>
          <p><strong>Predicted:</strong> {results.ethnicity.label}</p>
          <p><strong>Confidence:</strong> {(results.ethnicity.confidence * 100).toFixed(1)}%</p>
        </div>
      )}

      {results.emotion && (
        <div className="info-box">
          <h4>Emotion</h4>
          <p><strong>Predicted:</strong> {results.emotion.label}</p>
          <p><strong>Confidence:</strong> {(results.emotion.confidence * 100).toFixed(1)}%</p>
        </div>
      )}

      {!results.ethnicity && !results.emotion && (
        <div className="warning-box">
          <p>ℹ️ Additional models (Ethnicity, Emotion) are optional and may not be available.</p>
        </div>
      )}
    </div>
  )
}

