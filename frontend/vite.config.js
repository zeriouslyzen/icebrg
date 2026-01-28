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
    },
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        // If request has no extension and is not a directory, try adding .html
        if (!req.url.includes('.') && !req.url.endsWith('/')) {
          const url = req.url.split('?')[0];
          req.url = url + '.html';
        }
        next();
      });
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
        app: fileURLToPath(new URL('./app.html', import.meta.url)),
        encyclopedia: fileURLToPath(new URL('./encyclopedia.html', import.meta.url)),
        wiki: fileURLToPath(new URL('./wiki.html', import.meta.url)),
        features: fileURLToPath(new URL('./features.html', import.meta.url)),
        research: fileURLToPath(new URL('./research.html', import.meta.url)),
        study: fileURLToPath(new URL('./study.html', import.meta.url)),
        protocols: fileURLToPath(new URL('./protocols.html', import.meta.url)),
        admin: fileURLToPath(new URL('./admin.html', import.meta.url)),
        pegasus: fileURLToPath(new URL('./pegasus.html', import.meta.url)),
        dossier: fileURLToPath(new URL('./dossier.html', import.meta.url)),
        civilization: fileURLToPath(new URL('./civilization.html', import.meta.url)),
        matrix: fileURLToPath(new URL('./matrix.html', import.meta.url)),
        entity: fileURLToPath(new URL('./entity.html', import.meta.url)),
        investigations: fileURLToPath(new URL('./investigations.html', import.meta.url)),
        colossus_index: fileURLToPath(new URL('./colossus/index.html', import.meta.url)),
        colossus_graph: fileURLToPath(new URL('./colossus/graph.html', import.meta.url)),
        colossus_entity: fileURLToPath(new URL('./colossus/entity.html', import.meta.url))
      }
    }
  },
});
