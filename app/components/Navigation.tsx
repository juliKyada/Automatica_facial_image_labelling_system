'use client'

import styles from './Navigation.module.css'

interface NavigationProps {
  activeTab: string
  setActiveTab: (tab: string) => void
}

export default function Navigation({ activeTab, setActiveTab }: NavigationProps) {
  return (
    <nav className={styles.nav}>
      <button
        className={`${styles.navButton} ${activeTab === 'single' ? styles.active : ''}`}
        onClick={() => setActiveTab('single')}
      >
        Single Image
      </button>
      <button
        className={`${styles.navButton} ${activeTab === 'batch' ? styles.active : ''}`}
        onClick={() => setActiveTab('batch')}
      >
        Batch Processing
      </button>
      <button
        className={`${styles.navButton} ${activeTab === 'about' ? styles.active : ''}`}
        onClick={() => setActiveTab('about')}
      >
        About
      </button>
    </nav>
  )
}

