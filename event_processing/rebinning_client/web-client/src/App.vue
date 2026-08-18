<template>
  <div class="container-fluid d-flex flex-column vh-100">
    <!-- Navbar -->
    <nav class="navbar navbar-expand-lg navbar-light bg-light">
      <div class="container-fluid">
        <a class="navbar-brand" href="#">
          <img src="./assets/chrns-3-smaller.png" alt="logo" style="width: 16em;" class="d-inline-block align-text-top" />
          CHRNS event rebinning
        </a>
      </div>
    </nav>

    <!-- Main content -->
    <main class="flex-fill overflow-auto">
      <rebin-demo-page />
    </main>
  </div>
</template>

<script setup lang="ts">
import 'bootstrap/dist/css/bootstrap.min.css';
import RebinDemoPage from './pages/RebinDemoPage.vue';
</script>
