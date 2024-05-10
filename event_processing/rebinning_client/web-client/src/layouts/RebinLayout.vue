<template>
  <q-layout view="lHh Lpr lFf">
    <q-header elevated>
      <q-toolbar>
        <q-img fit="scale-down" src="/src/assets/chrns-3-smaller.png" style="width: 16em;"></q-img>
        <q-toolbar-title class="text-h4">CHRNS event rebinning</q-toolbar-title>

        <!-- <q-btn flat round dense icon="whatshot" /> -->
      </q-toolbar>
    </q-header>

    <q-page-container>
      <router-view />
    </q-page-container>
  </q-layout>
</template>

<script setup lang="ts">

</script>

<style>
body, #q-app {
  height: 100%;
}
</style>
