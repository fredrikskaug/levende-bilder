import { defineConfig } from 'vite';
import basicSsl from '@vitejs/plugin-basic-ssl';

// Dev server only – web/ is plain static files and can be deployed as-is.
// `npm run dev`        – http://localhost:5174 on this machine
// `npm run dev:https`  – https on the LAN, needed for the gyroscope on phones
//                        (accept the self-signed certificate warning once)
export default defineConfig(({ mode }) => ({
  root: 'web',
  publicDir: false,
  plugins: mode === 'https' ? [basicSsl()] : [],
  server: {
    port: 5174,
    strictPort: true, // don't drift to 5175, that's tools/server.py
    host: mode === 'https',
    // tools/server.py: collection search + depth maps on demand («Hent fra samlingen»)
    proxy: { '/api': 'http://127.0.0.1:5175' },
  },
}));
