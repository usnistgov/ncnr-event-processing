<template>
  <div class="container py-3">
    <div class="row align-items-center mb-3">
      <div class="col-auto">
        <label class="form-label">Num. Bins</label>
        <input class="form-control mx-2" type="number" v-model.number="num_bins" :disabled="!use_num" @keydown.enter="update_summary" />
      </div>
      <div class="col-auto">
        <label class="form-label">Mode</label>
        <select class="form-select mx-2" v-model="use_num">
          <option :value="true">Num.</option>
          <option :value="false">Width</option>
        </select>
      </div>
      <div class="col-auto">
        <label class="form-label">Bin Width</label>
        <input class="form-control mx-2" type="number" v-model.number="bin_width" :disabled="use_num" @keydown.enter="update_summary" />
      </div>
      <div class="col-auto">
        <label class="form-label">Start</label>
        <input class="form-control mx-2" type="number" v-model.number="rebin_limits.x1" />
      </div>
      <div class="col-auto">
        <label class="form-label">End</label>
        <input class="form-control mx-2" type="number" v-model.number="rebin_limits.x2" />
      </div>
      <div class="col-auto">
        <button class="btn btn-primary mx-2" @click="reset_start_end">Reset start + end</button>
      </div>
    </div>
    <div class="row justify-content-center mb-3">
      <div class="col-auto" v-if="selected_filename && selected_path">
        <button class="btn btn-success me-2" :disabled="fetching_summary" @click="update_summary">
          Show Summary
          <span v-if="fetching_summary" class="spinner-border spinner-border-sm ms-2" role="status" aria-hidden="true"></span>
        </button>
        <button class="btn btn-secondary" :disabled="downloading" @click="download_rebinned">
          Rebin + Download
          <span v-if="downloading" class="spinner-border spinner-border-sm ms-2" role="status" aria-hidden="true"></span>
        </button>
      </div>
      <form :action="`${rebinning_api}timebin/nexus_download`" method="post" target="hiddenFrame" class="d-none">
        <input ref="download_request_input" type="text" name="request_str" />
        <input ref="download_id_input" type="text" name="download_id" />
        <button ref="download_button" type="submit">Rebin + Download</button>
      </form>
      <iframe name="hiddenFrame" width="0" height="0" style="display:none;"></iframe>
    </div>
    <div class="row" v-show="shown_summary == selected_filename">
      <div class="col" ref="summary_plot_div"></div>
      <div class="col" ref="frame_plot_div"></div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, shallowRef, watchEffect } from 'vue';
import { react } from 'plotly.js-dist';
import { xSliceInteractor } from 'plotly-interactors';
import { api_get, api_post, rebinning_api, metadata, metadata_request, selected_filename, selected_path, rebin_limits } from '@/store';
import { NumpyArray, NestedArray } from '@/numpy_array';
import { setupDrawLoop } from '@/setupDrawLoop';
import type { TimeBins, SummaryTimeRequest } from '@/store';
import { v4 as uuidv4 } from 'uuid';

const download_button = ref<HTMLFormElement>();
const download_request_input = ref<HTMLInputElement>();
const download_id_input = ref<HTMLInputElement>();
const num_bins = ref(100);
const bin_width = ref(10);
const use_num = ref(true);
const x_slice_interactor = ref<xSliceInteractor>();

const shown_summary = ref('');
const fetching_summary = ref(false);
const downloading = ref(false);

const stored_bins = shallowRef<TimeBins>();

const summary_plot_div = ref<HTMLDivElement>();
const frame_plot_div = ref<HTMLDivElement>();

const summary_fig_template = {
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

const frame_fig_template = {
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
  const steps = Math.ceil((end - start) / step);
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
  const start = rebin_limits.x1;

  // 2. Find the first bin edge near 0 that aligns with x1
  const offset_start = start % bin_width.value;
  const offset_duration = metadata.value.duration - offset_start;
  const edges = get_edges(offset_duration, offset_start, offset_duration, bin_width.value, 0, false);

  return {
    mode: 'time',
    mask: null,
    edges: NumpyArray.from_array(edges)
  };
}

function get_rebin_time_bins_object() {
  const edges = get_edges(metadata.value.duration, rebin_limits.x1, rebin_limits.x2, bin_width.value, num_bins.value, use_num.value);
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
    while (!download_status?.complete && !download_status?.error) {
      await sleep(200);
      console.log('sleep awaited');
      download_status = await api_get(rebinning_api, `timebin/nexus_download_status/${download_id}`);
      console.log({download_status});
    }
    downloading.value = false;
    if (download_status?.error) {
      alert(`Error during rebinning: ${download_status.error}`);
    }
  }
}

async function update_summary() {
  if (metadata.value == null || metadata_request.value == null) {
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

  let summary;
  try {
    summary = await api_post(rebinning_api, 'summary_time', request_object);
  }
  catch (e) {
    alert(`Error fetching summary: ${e}`);
    return;
  }
  finally {
    fetching_summary.value = false;
  }

  shown_summary.value = selected_filename.value;

  const time_bin_edges = summary.bins.edges;
  const x = new NumpyArray(time_bin_edges).to_array();
  let y_min_max = [-Infinity, Infinity];
  const traces = Object.entries(summary.counts).map(([det_name, counts_obj]) => {
    const y = new NumpyArray(counts_obj).to_array();
    y_min_max = min_max(y, y_min_max);
    y.push(y.at(-1));
    return {x, y, name: det_name, line: {shape: 'hv'}}
  });
  const summary_fig = structuredClone(summary_fig_template);
  summary_fig.data = traces;

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

async function fetch_and_draw_frame({ det_name, point_number }: { det_name: string, point_number: number }) {
  const request_object: SummaryTimeRequest = {
    measurement: metadata_request.value,
    bins: stored_bins.value,
  }
  const frame_reply = await api_post(rebinning_api, `timebin/frame/${point_number}`, request_object);
  const frame_data = new NumpyArray(frame_reply.data[det_name]);
  // remove the first dimension, which is the time dimension
  frame_data.shape.splice(0, 1);
  const trace = { z: frame_data.to_array(), type: 'heatmap', transpose: true }

  const bins_array = new NumpyArray(stored_bins.value.edges).to_array();
  const start_time = bins_array[point_number];
  const end_time = bins_array[point_number + 1]
  const frame_fig = structuredClone(frame_fig_template);
  frame_fig.data = [trace];
  frame_fig.layout.title =  `Frame ${det_name}: ${start_time.toFixed(4)} < time < ${end_time.toFixed(4)} (s)`;
  react(frame_plot_div.value, frame_fig.data, frame_fig.layout, frame_fig.config);
}

const draw_loop = setupDrawLoop(fetch_and_draw_frame, 'draw frame');

function handle_summary_click(ev: { points: { data: { name: string }, pointNumber: number }[] }) {
  const { points } = ev;
  console.log({ev, points});
  if (points.length > 0) {
    const { data: { name: det_name }, pointNumber: point_number } = points[0];
    draw_loop.draw_requested.value = {det_name, point_number};
  }
}

async function reset_start_end() {
  rebin_limits.x1 = 0;
  rebin_limits.x2 = metadata.value.duration;
}

watchEffect(() => {
  const [_start, _end] = get_start_end(metadata.value.duration, rebin_limits.x1, rebin_limits.x2);

  if (use_num.value && num_bins.value != null && num_bins.value > 0) {
    bin_width.value = ( _end - _start ) / ( num_bins.value );
  }
  else if (bin_width.value != null && bin_width.value > 0) {
    num_bins.value = Math.ceil( ( _end - _start ) / bin_width.value );
  }

  if (x_slice_interactor.value?.update) {
    x_slice_interactor.value.update();
  }
});

onMounted(() => {
  console.log({react});
  const summary_fig = structuredClone(summary_fig_template);
  react(summary_plot_div.value, summary_fig.data, summary_fig.layout, summary_fig.config).then((splot) => {
    splot.on('plotly_click', handle_summary_click);
    splot.on('plotly_hover', handle_summary_click);
  });
  x_slice_interactor.value = new xSliceInteractor(rebin_limits, summary_plot_div.value, 'xy');
  const frame_fig = structuredClone(frame_fig_template);
  react(frame_plot_div.value, frame_fig.data, frame_fig.layout, frame_fig.config);
})
</script>

<style>
</style>
