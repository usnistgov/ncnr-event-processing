<template>
  <div class="col column q-px-md">
    <div class="row flex items-center">
      <q-input
        class="q-mx-md"
        v-model.number="num_bins"
        type="number"
        label="Num. Bins"
        :disable="!use_num"
      />
      <q-btn-toggle
        v-model="use_num"
        push
        toggle-color="primary"
        :options="[
          {label: 'Num.', value: true},
          {label: 'Width', value: false},
        ]"
      />
      <q-input
        class="q-mx-md"
        v-model.number="bin_width"
        type="number"
        label="Bin Width"
        :disable="use_num"
      />
      <q-input
        class="q-mx-md"
        v-model.number="start"
        type="number"
        label="Start"
        :disable="use_num"
      />
      <q-input
        class="q-mx-md"
        v-model.number="end"
        type="number"
        label="End"
        :disable="use_num"
      />
      <q-btn label="Reset start + end" color="primary"></q-btn>
    </div>
    <div class="col column">
      <div class="col row">
        <div class="plotly col" ref="summary_plot_div" style=""></div>
        <div class="plotly col" ref="frame_plot_div"></div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue';
import { react } from 'plotly.js-dist'


const num_bins = ref(100);
const bin_width = ref(10);
const use_num = ref(true);

const start = ref(0);
const end = ref(3600);

const summary_plot_div = ref<HTMLDivElement>();
const frame_plot_div = ref<HTMLDivElement>();

const summary_fig = {
    'data': [],
    'layout': {
        title: 'Time Summary',
        // margin=dict(l=0, r=0, t=30, b=0),
        xaxis: { title: 'elapsed time (seconds)', showline: true, mirror: true, showgrid: true },
        yaxis: { title: 'total counts', showline: true, mirror: true, showgrid: true },
        template: 'simple_white',
    },
    'config': {responsive: true},
}

const frame_fig = {
    'data': [],
    'layout': {
        title: 'Frame snapshot',
        xaxis: { showline: true, mirror: true, showgrid: true },
        yaxis: { showline: true, mirror: true, showgrid: true, scaleanchor: 'x', scaleratio: 1 },
        template: 'simple_white',
    },
    'config': {responsive: true},
}


onMounted(() => {
  console.log({react});
  react(summary_plot_div.value, summary_fig.data, summary_fig.layout, summary_fig.config);
  react(frame_plot_div.value, frame_fig.data, frame_fig.layout, frame_fig.config);
})

</script>

<style>
</style>
