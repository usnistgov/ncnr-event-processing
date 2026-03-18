<template>
  <div class="container py-3">
    <div class="row mb-3">
      <div class="col-auto">
        <h5 class="mb-2">Search:</h5>
        <div class="mb-2">
          <label class="form-label">Instrument Name</label>
          <select class="form-select" v-model="search_inputs.instrument_name" @change="search">
            <option value="">All</option>
            <option v-for="inst in instrument_names" :key="inst" :value="inst">{{ inst }}</option>
          </select>
        </div>
        <div class="mb-2">
          <label class="form-label">Participant Name</label>
          <input class="form-control" v-model="search_inputs.participant_name" @input="search" />
        </div>
        <div class="mb-2">
          <label class="form-label">Title</label>
          <input class="form-control" v-model="search_inputs.experiment_title" @input="search" />
        </div>
      </div>
      <div class="col">
        <table class="table table-striped table-hover">
          <thead>
            <tr>
              <th v-for="col in columns" :key="col.name">{{ col.label }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="row in rows" :key="row.id" @click="selectRow(row)" :class="{'table-active': selected.includes(row)}">
              <td>{{ row.id }}</td>
              <td>{{ row.title }}</td>
              <td>{{ Array.isArray(row.participant_names) ? row.participant_names.join(', ') : row.participant_names }}</td>
            </tr>
          </tbody>
        </table>
            <PaginationControls
  :rows-per-page="pagination.rowsPerPage"
  :page="pagination.page"
  :rows-number="pagination.rowsNumber"
  @updateRowsPerPage="updateRowsPerPage"
  @goToPage="goToPage"
/>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed } from 'vue';
import PaginationControls from './PaginationControls.vue';
import { api_get, ncnr_metadata_api, all_instruments, selected_experiment, active_tab } from '@/store';

const endpoint = 'experiments';
const columns = [
    { name: 'experiment_id', label: 'Experiment ID', field: 'id', required: true, align: 'left', headerStyle: { width: '10em' } },
    { name: 'title', label: 'Title', field: 'title', align: 'left' },
    { name: 'participants', label: 'Participants', field: 'participant_names', align: 'left', format: (value: string) => JSON.parse(value).join(', '), style: 'min-width:2px;' },
];

interface Row {
  id: string;
  title: string;
  participant_names: string[];
}

interface APISearchParams {
  offset: number;
  limit: number;
  instrument_id?: string;
  participant_name?: string;
  title?: string;
  id?: string;
  full_count?: boolean;
}

const rows = ref<Row[]>([]);
const selected = ref<Row[]>([]);
const pagination = reactive({ rowsPerPage: 10, descending: true, sortBy: 'date', page: 1, rowsNumber: 0 });

// computed total pages
const totalPages = computed(() => {
  return pagination.rowsNumber > 0 ? Math.ceil(pagination.rowsNumber / pagination.rowsPerPage) : 1;
});

function updateRowsPerPage() {
  pagination.page = 1; // reset to first page when rows per page changes
  search(false);
}

function goToPage(newPage: number) {
  if (newPage < 1) newPage = 1;
  if (newPage > totalPages.value) newPage = totalPages.value;
  pagination.page = newPage;
  search(false);
}

const search_inputs = ref({ instrument_name: '', experiment_title: '', experiment_id: '', participant_name: '' });

function on_selection(event: { keys: string[] }) {
  const { keys } = event;
  if (keys.length > 0) {
    selected_experiment.value = keys[0];
    active_tab.value = 'datafile_search';
  }
}

function row_dblclick(_evt: unknown, row: Row) {
  selected.value.splice(0, 1, row);
  selected_experiment.value = row.id;
}

function selectRow(row: Row) {
  on_selection({ keys: [row.id] });
}

const instrument_names = ['vsans', 'macs', 'candor'];

async function search(update_total = true) {
  const { rowsPerPage, page } = pagination;
  const offset = update_total ? 0 : rowsPerPage * (page - 1);
  const params: APISearchParams = { offset, limit: rowsPerPage };
  const { instrument_name, participant_name, experiment_id, experiment_title } = search_inputs.value;
  if (instrument_name) {
    const instrument_id = all_instruments.value.find((instr) => instr.alias === instrument_name)?.id;
    params.instrument_id = instrument_id;
  }
  if (participant_name) params.participant_name = `%${participant_name}%`;
  if (experiment_title) params.title = `%${experiment_title}%`;
  if (experiment_id) params.id = `${experiment_id}`;
  if (update_total) {
    const full_count_params = { ...params, full_count: true } as APISearchParams;
    const full_count_result = await api_get(ncnr_metadata_api, endpoint, full_count_params);
    if (full_count_result.length) {
      pagination.rowsNumber = full_count_result[0]?.full_count ?? 0;
      pagination.page = 1;
    }
  }
  const r = await api_get(ncnr_metadata_api, endpoint, params);
  rows.value = r;
}

async function pagination_request_handler(request: { pagination: { rowsPerPage: number; page: number } }) {
  pagination.rowsPerPage = request.pagination.rowsPerPage;
  pagination.page = request.pagination.page;
  await search(false);
}

</script>

