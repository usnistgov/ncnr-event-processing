<template>
  <q-layout view="lHh Lpr lFf" class="column no-wrap">
    <q-header elevated class="row">
      <q-toolbar>
        <q-img fit="scale-down" src="./assets/chrns-3-smaller.png" style="width: 16em;"></q-img>
        <q-toolbar-title class="text-h4">CHRNS event rebinning</q-toolbar-title>

        <!-- <q-btn flat round dense icon="whatshot" /> -->
      </q-toolbar>
    </q-header>

    <q-page-container class="row column col">
      <rebin-demo-page />
    </q-page-container>
  </q-layout>
</template>

<script setup lang="ts">
import RebinDemoPage from './pages/RebinDemoPage.vue';
</script>
