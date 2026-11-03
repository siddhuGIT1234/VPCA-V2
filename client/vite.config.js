// Dev server config.
//   npm run dev        -> http://localhost:5173        (desktop development)
//   npm run dev:https  -> https://<your-ip>:5173       (Quest 2 over Wi-Fi; VR needs HTTPS)
//
// The Python server (port 8000) is reached THROUGH this dev server:
//   /api/...  -> http://localhost:8000/...
//   /ws/...   -> ws://localhost:8000/...
// so the page and the data share one origin. That avoids mixed-content
// blocking on HTTPS and means only port 5173 has to be reachable from the Quest.
import { defineConfig } from 'vite';
import basicSsl from '@vitejs/plugin-basic-ssl';

const PY = 'localhost:8000';

export default defineConfig(({ mode }) => ({
  plugins: mode === 'https' ? [basicSsl()] : [],
  server: {
    host: true,
    port: 5173,
    proxy: {
      '/api': { target: `http://${PY}`, rewrite: (p) => p.replace(/^\/api/, '') },
      '/ws': { target: `ws://${PY}`, ws: true, rewrite: (p) => p.replace(/^\/ws/, '') },
    },
  },
}));
