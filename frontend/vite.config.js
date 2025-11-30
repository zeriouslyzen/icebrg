import { defineConfig } from 'vite';

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
    minify: 'terser'
  }
});

