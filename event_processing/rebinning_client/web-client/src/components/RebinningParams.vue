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
    <div class="row justify-center">
      <q-btn :disabled="fetching_summary" v-if="selected_filename && selected_path" style="height: 1em;" color="positive" @click="update_summary">
        <div class="row items-center no-wrap">
          <div class="text-center">
            Show Summary
          </div>
          <q-spinner size="1em" color="white" v-if="fetching_summary"></q-spinner>
        </div>
      </q-btn>
      <q-btn :disabled="downloading" v-if="selected_filename && selected_path" style="height: 1em;" class="q-mx-md" color="secondary" @click="download_rebinned" >
        <div class="row items-center no-wrap">
          <div class="text-center">
            Rebin + Download
          </div>
          <q-spinner size="1em" color="white" v-if="downloading"></q-spinner>
        </div>
      </q-btn>
      <form :action="`${rebinning_api}/timebin/nexus_download`" method="post" target="hiddenFrame">
        <input ref="download_request_input" type="text" style="display:none;" name="request_str" />
        <input ref="download_id_input" type="text" style="display:none;" name="download_id" />
        <button ref="download_button" type="submit" style="display:none;">Rebin + Download</button>
      </form>
      <iframe name="hiddenFrame" width="0" height="0" border="0" style="display:none;"></iframe>
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
import { ref, onMounted, shallowRef, watchEffect, toRaw } from 'vue';
import { react } from 'plotly.js-dist';
import { api_get, api_post, rebinning_api, metadata, metadata_request, selected_filename, selected_path } from 'src/store';
import { NumpyArray, NestedArray } from 'src/numpy_array';
import type { TimeBins, SummaryTimeRequest, MetadataRequest } from 'src/store';
import { v4 as uuidv4 } from 'uuid';
import { store } from 'quasar/wrappers';
import { isCallOrNewExpression } from 'typescript';

const download_button = ref<HTMLFormElement>();
const download_request_input = ref<HTMLInputElement>();
const download_id_input = ref<HTMLInputElement>();
const num_bins = ref(100);
const bin_width = ref(10);
const use_num = ref(true);

const fetching_summary = ref(false);
const downloading = ref(false);

function set_num_bins(value_str: string) {
  const new_value = parseFloat(value_str);
  if (!isNaN(new_value)) {
    console.log({new_value})
    num_bins.value = new_value;
  }
}

const start = ref(0);
const end = ref(3600);
const stored_bins = shallowRef<TimeBins>();

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


function arange(start: number, end: number, step: number = 1) {
  const steps = Math.trunc((end - start) / step);
  return Array.from({length: steps}).map((_, i) => start + (i * step));
}

function linspace(start: number, end: number, steps: number, endpoint: boolean = true) {
  const denom = (endpoint) ? (steps - 1) : steps;
  const step = (end - start) / denom;
  return Array.from({length: steps}).map((_, i) => start + (i * step));
}

function get_start_end(duration: number, nominal_start: number | null, nominal_end: number | null) {
  let start = nominal_start;
  let end = nominal_end;

  if (start == null) {
    start = 0;
  }
  else if (start < 0) {
    start += duration;
  }

  if (end == null) {
    end = duration;
  }
  else if (end < 0) {
    end += duration;
  }

  return [start, end];
}

function get_edges(duration: number, nominal_start: number | null, nominal_end: number | null, bin_width: number, num_bins: number, use_num: boolean = true) {
  const [start, end] = get_start_end(duration, nominal_start, nominal_end);
  let edges: number[];

  if (use_num) {
    edges = linspace(start, end, num_bins + 1);
  }
  else if (bin_width != null) {
    edges = arange(start, end + bin_width, bin_width);
    if (edges.at(-2) == duration) { // last bin is full so don't need a partial bin more
      edges = edges.slice(0, -1);
    }
  }
  else {
    throw new Error('Must specify one of interval or nbins for edges')
  }
  // console.log({duration, nominal_start, nominal_end, bin_width, num_bins, use_num, edges});
  return edges
}

function get_summary_time_bins_object() {
  const offset_start = start.value % bin_width.value;
  const edges = get_edges(metadata.value.duration, offset_start, metadata.value.duration, bin_width.value, num_bins.value, use_num.value);
  const result: TimeBins = {
    mode: 'time',
    mask: null,
    edges: NumpyArray.from_array(edges)
  }
  // console.log({summary_time_bins_object: result, edges})
  return result;
}

function get_rebin_time_bins_object() {
  const edges = get_edges(metadata.value.duration, start.value, end.value, bin_width.value, num_bins.value, use_num.value);
  const result: TimeBins = {
    mode: 'time',
    mask: null,
    edges: NumpyArray.from_array(edges)
  }
  return result;
}

function min_max(array: NestedArray<number>, start_value = [-Infinity, Infinity]) {
  const flattened = array.flat() as number[];
  return flattened.reduce(([a,z], v) => ([Math.min(a,v), Math.max(z,v)]), start_value);
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function download_rebinned() {
  const bins = get_rebin_time_bins_object();
  const request_object: SummaryTimeRequest = {
    measurement: metadata_request.value,
    bins
  };
  const download_id = uuidv4();
  const request_string = JSON.stringify(request_object);
  if (download_request_input?.value && download_id_input.value && download_button?.value) {
    downloading.value = true;
    download_request_input.value.value = request_string;
    download_id_input.value.value = download_id;
    console.log('pre-click...');
    download_button.value.click();
    console.log('post-click, about to get...');
    let download_status = await api_get(rebinning_api, `timebin/nexus_download_status/${download_id}`);
    console.log('status retrieved.');
    console.log({download_status});
    while (download_status) {
      await sleep(200);
      console.log('sleep awaited');
      download_status = await api_get(rebinning_api, `timebin/nexus_download_status/${download_id}`);
      console.log({download_status});
    }
    downloading.value = false;
  }
}

async function update_summary() {
  if (metadata.value == null) {
    alert('no file loaded');
    return;
  }

  const bins = get_summary_time_bins_object();
  stored_bins.value = bins;

  fetching_summary.value = true;
  const request_object: SummaryTimeRequest = {
    measurement: metadata_request.value,
    bins
  }

  const summary = await api_post(rebinning_api, 'summary_time', request_object);
  fetching_summary.value = false;

  const time_bin_edges = summary.bins.edges;
  const x = new NumpyArray(time_bin_edges).to_array();
  let y_min_max = [-Infinity, Infinity];
  const traces = Object.entries(summary.counts).map(([det_name, counts_obj]) => {
    const y = new NumpyArray(counts_obj).to_array();
    y_min_max = min_max(y, y_min_max);
    y.push(y.at(-1));
    return {x, y, name: det_name, line: {shape: 'hv'}}
  });
  summary_fig['data'] = traces;

  const [y_min, y_max] = y_min_max;
  const y_range = (y_max - y_min);
  const display_y_min = y_min - 0.1 * y_range;
  const display_y_max = y_max + 0.1 * y_range;
  const x_range = metadata.value.duration;
  const x_max = x_range;
  const x_min = 0.0;
  const display_x_min = x_min - 0.1 * x_range;
  const display_x_max = x_max + 0.1 * x_range;

  summary_fig.layout.yaxis.range = [display_y_min, display_y_max];
  summary_fig.layout.xaxis.range = [display_x_min, display_x_max];
  summary_fig.layout.xaxis.autorangeoptions = {maxallowed: display_x_max, minallowed: display_x_min};

  react(summary_plot_div.value, summary_fig.data, summary_fig.layout, summary_fig.config);
}

async function show_frame(det_name: string, point_number: number) {
  const request_object: SummaryTimeRequest = {
    measurement: metadata_request.value,
    bins: stored_bins.value,
  }
  const frame_reply = await api_post(rebinning_api, `timebin/frame/${point_number}`, request_object);
  const frame_data = new NumpyArray(frame_reply.data[det_name]);

  const x = linspace(0.5, frame_data.shape[0] + 0.5, frame_data.shape[0]);
  const y = linspace(0.5, frame_data.shape[1] + 0.5, frame_data.shape[1]);
  frame_data.shape.splice(2, 1);
  const trace = { z: frame_data.to_array(), type: 'heatmap' }

  const bins_array = new NumpyArray(stored_bins.value.edges).to_array();
  const start_time = bins_array[point_number];
  const end_time = bins_array[point_number + 1]
  frame_fig.data = [trace];
  frame_fig.layout.title =  `Frame ${det_name}: ${start_time.toFixed(4)} < time < ${end_time.toFixed(4)} (s)`;
  react(frame_plot_div.value, frame_fig.data, frame_fig.layout, frame_fig.config);

}

function handle_summary_click(ev) {
  const { points } = ev;
  if (points.length > 0) {
    const { data: { name }, pointNumber } = points[0];
    show_frame(name, pointNumber);

  }
}

watchEffect(() => {
  const [_start, _end] = get_start_end(metadata.value.duration, start.value, end.value);

  if (use_num.value && num_bins.value != null && num_bins.value > 0) {
    bin_width.value = ( _end - _start ) / ( num_bins.value );
  }
  else if (bin_width.value != null && bin_width.value > 0) {
    num_bins.value = Math.ceil( ( _end - _start ) / bin_width.value );
  }

});


onMounted(() => {
  console.log({react});
  react(summary_plot_div.value, summary_fig.data, summary_fig.layout, summary_fig.config).then((splot) => {
    splot.on('plotly_click', handle_summary_click);
    splot.on('plotly_hover', handle_summary_click);
  });
  react(frame_plot_div.value, frame_fig.data, frame_fig.layout, frame_fig.config);
})

</script>

<style>
</style>
