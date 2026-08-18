import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';

export default defineConfig({
  base: '', // relative asset paths
  plugins: [vue()],
  resolve: {
    alias: {
      '@': '/src'
    }
  }
});