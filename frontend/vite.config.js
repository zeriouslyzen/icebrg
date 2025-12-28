import { defineConfig } from 'vite';
import { fileURLToPath } from 'url';

export default defineConfig({
  server: {
    port: 3000,
    host: '0.0.0.0',
    strictPort: false,
    open: false,
    cors: true,
    hmr: false,
    watch: {
      ignored: ['**/*.md', '**/node_modules/**']
    },
    fs: {
      allow: ['..']
    }
  },
  publicDir: 'public', // Include public directory with research data
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    sourcemap: false,
    minify: 'terser',
    rollupOptions: {
      input: {
        main: fileURLToPath(new URL('./index.html', import.meta.url)),
        encyclopedia: fileURLToPath(new URL('./encyclopedia.html', import.meta.url)),
        wiki: fileURLToPath(new URL('./wiki.html', import.meta.url)),
        features: fileURLToPath(new URL('./features.html', import.meta.url)),
        research: fileURLToPath(new URL('./research.html', import.meta.url)),
        study: fileURLToPath(new URL('./study.html', import.meta.url)),
        protocols: fileURLToPath(new URL('./protocols.html', import.meta.url))
      }
    }
  },
});
