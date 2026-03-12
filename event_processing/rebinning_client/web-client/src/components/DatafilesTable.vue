<template>
  <div class="container py-3">
    <div class="row mb-3">
      <div class="col-auto">
        <h5 class="mb-2">Search:</h5>
        <div class="mb-2">
          <label class="form-label">Filename</label>
          <input class="form-control" v-model="search_inputs.filename" @input="search" />
        </div>
        <div class="mb-2">
          <label class="form-label">Description (% wild)</label>
          <input class="form-control" v-model="search_inputs.description" @input="search" />
        </div>
      </div>
      <div class="col">
        <table class="table table-striped table-hover">
          <thead>
            <tr>
              <th v-for="col in all_columns" :key="col.name">{{ col.label }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="row in rows_with_metadata" :key="row.filename" @click="selectRow(row)" :class="{'table-active': selected.includes(row)}">
              <td v-for="col in all_columns" :key="col.name">
                {{ row[col.field as any] ?? row.metadata?.[col.name] ?? '' }}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue';
import { api_get, ncnr_metadata_api, selected_experiment, selected_filename, selected_path, get_metadata, active_tab } from '@/store';

const endpoint = 'datafiles';
const columns: {name: string, label: string, field?: string, required?: boolean, metadata?: string, ':field'?: string }[] = [
  { name: 'filename', label: 'Filename', field: 'filename', required: true },
  { name: 'cycle', label: 'Rx Cycle', field: 'rxcycle_id' },
  { name: 'start_date', label: 'Start Date', field: 'start_date' },
];

const rows_with_metadata = computed(() => rows.value.map(row => ({
  ...row,
  metadata: row.metadata ? JSON.parse(row.metadata) : {}
})));

const all_columns = computed(() => {
  const extra: typeof columns = [];
  const keys = new Set<string>();
  for (const r of rows_with_metadata.value) {
    for (const k in r.metadata) keys.add(k);
  }
  for (const k of keys) {
    extra.push({ name: k, label: k, field: k });
  }
  return columns.concat(extra);
});

type Row = { filename: string; rxcycle_id: string; start_date: string; localdir: string; metadata?: string };
const rows = ref<Row[]>([]);
const selected = ref<Row[]>([]);
const pagination = ref({ rowsPerPage: 10, descending: true, sortBy: 'date', page: 1, rowsNumber: 0 });

const search_inputs = ref({ filename: '', description: '' });

async function on_selection(ev: { added: boolean; rows: Row[] }) {
  const { added, rows: evRows } = ev;
  if (added) {
    const { filename, localdir } = evRows[0];
    selected_filename.value = filename;
    selected_path.value = localdir;
    await get_metadata();
    active_tab.value = 'rebinning_params';
    selected.value.splice(0, 1, evRows[0]);
  } else {
    selected.value = [];
    selected_filename.value = '';
    selected_path.value = '';
  }
}

function selectRow(row: Row) {
  on_selection({ added: true, rows: [row] });
}

async function search(update_total = true) {
  const { rowsPerPage, page } = pagination.value;
  const offset = update_total ? 0 : rowsPerPage * (page - 1);
  const params: Record<string, any> = { offset, limit: rowsPerPage, experiment_id: selected_experiment.value };
  const { filename } = search_inputs.value;
  if (filename) params.filename = `%${filename}%`;
  const { description } = search_inputs.value;
  if (description) params.metadata = JSON.stringify([{ property_path: ['description'], comparison: 'like', value: description }]);
  if (update_total) {
    const fullCount = await api_get(ncnr_metadata_api, endpoint, { ...params, full_count: true });
    if (fullCount.length) {
      pagination.value.rowsNumber = fullCount[0]?.full_count ?? 0;
      pagination.value.page = 1;
    }
  }
  rows.value = await api_get(ncnr_metadata_api, endpoint, params);
}

watch(() => selected_experiment.value, () => { search(); }, { immediate: true });
</script>
