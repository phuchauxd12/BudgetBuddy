const path = require('path');
const vuePlugin = require('@vitejs/plugin-vue');

const { defineConfig } = require('vite');

/**
 * https://vitejs.dev/config
 */
const config = defineConfig({
  root: path.join(__dirname, 'src', 'renderer'),
  publicDir: 'public',
  server: {
    port: 8081,
  },
  open: false,
  build: {
    outDir: path.join(__dirname, 'build', 'renderer'),
    emptyOutDir: true,
    target: 'esnext',
  },
  plugins: [vuePlugin()],
  pluginOptions: {
    electronBuilder: {
      nodeIntegration: false,
    },
  },

  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src/@"),
    },
  },
});

module.exports = config;
