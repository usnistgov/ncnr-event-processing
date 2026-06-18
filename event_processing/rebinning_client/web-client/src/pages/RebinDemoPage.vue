<template>
  <div class="container-fluid">
    <!-- Nav tabs -->
    <ul class="nav nav-tabs mb-3">
      <li class="nav-item">
        <a class="nav-link" :class="{ active: active_tab === 'experiment_search' }" href="#" @click.prevent="active_tab = 'experiment_search'">Select Experiment</a>
      </li>
      <li class="nav-item">
        <a class="nav-link" :class="{ active: active_tab === 'datafile_search' }" href="#" @click.prevent="active_tab = 'datafile_search'">Select File</a>
      </li>
      <li class="nav-item">
        <a class="nav-link" :class="{ active: active_tab === 'rebinning_params' }" href="#" @click.prevent="active_tab = 'rebinning_params'">Rebinning Params</a>
      </li>
    </ul>

    <hr class="my-3" />

    <div class="row gx-3 align-items-center mb-3">
      <div class="col">
        <label class="form-label">Experiment ID</label>
        <input class="form-control" v-model="selected_experiment" readonly />
      </div>
      <div class="col">
        <label class="form-label">Filename</label>
        <input class="form-control" v-model="selected_filename" readonly />
      </div>
      <div class="col" style="max-width: 20em;">
        <label class="form-label">File Path</label>
        <input class="form-control" v-model="selected_path" readonly />
      </div>
      <div class="col">
        <label class="form-label">Duration</label>
        <input class="form-control" v-model="metadata.duration" readonly />
      </div>
    </div>

    <!-- Tab panels -->
    <div v-show="active_tab === 'experiment_search'" class="mb-3">
      <experiments-table />
    </div>
    <div v-show="active_tab === 'datafile_search'" class="mb-3">
      <datafiles-table />
    </div>
    <div v-show="active_tab === 'rebinning_params'" class="mb-3">
      <keep-alive>
        <rebinning-params class="d-flex flex-column" />
      </keep-alive>
    </div>
  </div>
</template>

<script setup lang="ts">
// import ExampleComponent from 'components/ExampleComponent.vue';
import ExperimentsTable from '@/components/ExperimentsTable.vue';
import DatafilesTable from '@/components/DatafilesTable.vue';
import RebinningParams from '@/components/RebinningParams.vue';

import { selected_experiment, selected_filename, selected_path, metadata, active_tab } from '@/store';

</script>

<style>
.q-tab-panels > .q-panel {
  display: flex;
  flex-direction: column;
  flex: 1 1 0;
}
</style>
