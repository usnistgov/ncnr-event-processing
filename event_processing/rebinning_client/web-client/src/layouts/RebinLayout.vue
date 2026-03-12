<template>
  <div class="container-fluid d-flex flex-column vh-100">
    <!-- Navbar -->
    <nav class="navbar navbar-expand-lg navbar-light bg-light">
      <div class="container-fluid">
        <a class="navbar-brand" href="#">
          <img src="/src/assets/chrns-3-smaller.png" alt="logo" style="width: 16em;" class="d-inline-block align-text-top" />
          CHRNS event rebinning
        </a>
      </div>
    </nav>

    <!-- Main content -->
    <main class="flex-fill overflow-auto">
      <router-view />
    </main>
  </div>
</template>

<script setup lang="ts">

</script>

<style>
body, #q-app {
  height: 100%;
}
</style>
