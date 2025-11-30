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
    // HMR will automatically use the request origin, so it works for both localhost and network access
    hmr: {
      clientPort: 3000 // Use the same port for HMR
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
        encyclopedia: resolve(__dirname, 'encyclopedia.html')
      }
    }
  },
  plugins: [
    {
      name: 'copy-encyclopedia',
      closeBundle() {
        // Copy encyclopedia.html to dist after build
        try {
          copyFileSync(
            resolve(__dirname, 'encyclopedia.html'),
            resolve(__dirname, 'dist', 'encyclopedia.html')
          );
        } catch (error) {
          console.warn('Could not copy encyclopedia.html:', error);
        }
      }
    }
  ]
});
