import { defineConfig } from 'vite';
import { copyFileSync } from 'fs';
import { resolve } from 'path';

export default defineConfig({
  server: {
    port: 3000,
    host: '0.0.0.0', // Explicitly bind to all interfaces for network access
    strictPort: false,
    open: false, // Don't auto-open browser
    cors: true, // Enable CORS
    // HMR disabled to prevent constant refreshing during development
    hmr: false,
    watch: {
      ignored: ['**/*.md', '**/node_modules/**'] // Ignore markdown files too
    }
  },
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    sourcemap: false,
    minify: 'terser',
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        encyclopedia: resolve(__dirname, 'encyclopedia.html'),
        wiki: resolve(__dirname, 'wiki.html'),
        features: resolve(__dirname, 'features.html'),
        research: resolve(__dirname, 'research.html')
      }
    }
  },
});
