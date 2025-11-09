import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  preview: {
    host: '0.0.0.0',
    port: 4173,
    allowedHosts: [
      'ar-focus-app-te.onrender.com',
      'ar-focus-app-3.onrender.com',
      'ar-focus-app-2.onrender.com',
      'ar-focus-app.onrender.com',
      'ar-focus-app-beta.onrender.com',
    ],
  },
})
