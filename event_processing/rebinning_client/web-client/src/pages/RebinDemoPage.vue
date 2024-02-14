<template>
  <q-page class="full-height column">
    <q-tabs
      class="w-full"
      v-model="active_tab"
      dense
      >
      <q-tab name="experiment_search" label="Select Experiment" key="experiment_search"></q-tab>
      <q-tab name="datafile_search" label="Select File" key="datafile_search"></q-tab>
      <q-tab name="rebinning_params" label="Rebinning Params" key="rebinning_params"></q-tab>
    </q-tabs>
    <q-separator/>
    <div class="row q-px-lg items-center">
      <q-input class="q-px-md" v-model="selected_experiment" readonly label="Experiment ID"></q-input>
      <q-input class="q-px-md" v-model="selected_filename" readonly label="Filename"></q-input>
      <q-input class="q-px-md" v-model="selected_path" readonly label="File Path" :input-style="{width:'20em'}"></q-input>
      <q-input class="q-px-md" v-model="metadata.duration" readonly label="Duration"></q-input>
      <!-- <q-btn v-if="selected_filename && selected_path" style="height: 1em;" color="positive" label="Show Summary" />
      <q-btn v-if="selected_filename && selected_path" style="height: 1em;" class="q-mx-md" color="secondary" label="Rebin + Download" /> -->
    </div>
    <!-- <q-separator/> -->
    <q-tab-panels v-model="active_tab" class="column col q-py-sm " animated keep-alive>
      <q-tab-panel class="q-px-md" name="experiment_search" key="experiment_search">
          <experiments-table />
      </q-tab-panel>
      <q-tab-panel name="datafile_search" key="datafile_search">
        search files
        <datafiles-table />
      </q-tab-panel>
      <q-tab-panel class="column col q-px-md" name="rebinning_params" key="rebinning_params">
        <rebinning-params class="column col"/>
      </q-tab-panel>
    </q-tab-panels>
  </q-page>
</template>

<script setup lang="ts">
// import ExampleComponent from 'components/ExampleComponent.vue';
import ExperimentsTable from 'components/ExperimentsTable.vue';
import DatafilesTable from 'src/components/DatafilesTable.vue';
import RebinningParams from 'src/components/RebinningParams.vue';

import { selected_experiment, selected_filename, selected_path, metadata, active_tab } from 'src/store';

</script>

<style>
.q-tab-panels > .q-panel {
  display: flex;
  flex-direction: column;
  flex: 1 1 0;
}
</style>
