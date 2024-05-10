<template>
  <div class="col column q-pa-md">

      <div class="row col">
        <div class="col-auto">
          <div class="text-h6 q-py-sm">Search:</div>
          <q-input
            v-model="search_inputs.filename"
            debounce="500"
            label="Filename"
            clearable
            @update:model-value="search"
          ></q-input>
          <q-input
            v-model="search_inputs.description"
            debounce="500"
            label="Description (% wild)"
            clearable
            @update:model-value="search"
          ></q-input>
        </div>
        <div class="col">
          <q-table
            class="sticky-header-table-datafiles my-sticky-column-table"
            flat bordered
            :rows="rows_with_metadata"
            :columns="all_columns"
            row-key="filename"
            selection="single"
            v-model:selected="selected"
            @selection="on_selection"
            @row-dblclick="row_dblclick"
            v-model:pagination="pagination"
            @request="pagination_request_handler"
            /> 
        </div>
      </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue';
import { api_get, ncnr_metadata_api, selected_experiment, selected_filename, selected_path, get_metadata, active_tab } from 'src/store';

const endpoint = 'datafiles';
const columns: {name: string, label: string, field?: string, required?: boolean, metadata?: string, ':field'?: string }[] = [
  { 'name': 'filename', 'label': 'Filename', 'field': 'filename', 'required': true },
  { 'name': 'cycle', 'label': 'Rx Cycle', 'field': 'rxcycle_id' },
  { 'name': 'start_date', 'label': 'Start Date', 'field': 'start_date' },
]

const rows_with_metadata = computed(() => {
  return rows.value.map(row => {
    const metadata = row.metadata ? JSON.parse(row.metadata) : {};
    return {...row, metadata};
  });
});

const all_columns = computed(() => {
  const keys = new Set<string>();
  for (const row of rows_with_metadata.value) {
    for (const key in row.metadata) {
      keys.add(key);
    }
  }

  const extra_cols = [];
  for (const key of keys) {
    const col_def = {'name': key, 'label': key, field: (row: Row) => row?.metadata?.[key]};
    extra_cols.push(col_def);
  }
  return columns.concat(extra_cols);
})

interface APISearchParams {
  offset: number,
  limit: number,
  experiment_id: string,
  filename?: string,
  full_count?: boolean,
  metadata?: string,
}

type Row = {filename: string, rxcycle_id: string, start_date: string, localdir: string, metadata?: string};
const rows = ref<Row[]>([]);
const selected = ref<Row[]>([]);
const pagination = ref({
  'rowsPerPage': 10,
  'descending': true,
  'sortBy': 'date',
  'page': 1,
  'rowsNumber': 0,
})

const search_inputs = ref({
  filename: '',
  description: ''
});

async function on_selection(ev: { added: boolean, rows: Row[] }) {
  const { added, rows } = ev;
  if (added) {
    const { filename, localdir } = rows[0];
    selected_filename.value = filename;
    selected_path.value = localdir;
    await get_metadata();
    active_tab.value = 'rebinning_params';
  }
  else {
    selected_filename.value = selected_path.value = '';
  }
}

function row_dblclick(_evt: unknown, row: Row) {
  selected.value.splice(0, 1, row);
  on_selection({ added: true, rows: [row]});
}

async function search(update_total = true) {
  // update_total is True for new searches, but not when paginating
  // reset offset for new searches:
  const { rowsPerPage, page } = pagination.value;
  const offset = (update_total) ? 0 : rowsPerPage * (page - 1);
  const params: APISearchParams = { offset, limit: pagination.value.rowsPerPage, experiment_id: selected_experiment.value }
  const { filename } = search_inputs.value;
  if (filename) {
    params['filename'] = `%${filename}%`;
  }
  const { description } = search_inputs.value;
  if (description) {
    params['metadata'] = JSON.stringify([{property_path: ['description'], comparison: 'like', value: `${description}` }]);
  }
  if (update_total) {
    const full_count_params: APISearchParams = {'full_count': true, ...params};
    const full_count_result = await api_get(ncnr_metadata_api, endpoint, full_count_params);
    if (full_count_result.length > 0) {
      const rowsNumber = full_count_result[0]?.full_count ?? 0;
      pagination.value['rowsNumber'] = rowsNumber;
      pagination.value['page'] = 1;
    }
  }

  const r = await api_get(ncnr_metadata_api, endpoint, params);
  rows.value = r;
}

async function pagination_request_handler(request: {pagination: { rowsPerPage: number, page: number }}) {
  pagination.value.rowsPerPage = request.pagination.rowsPerPage;
  pagination.value.page = request.pagination.page;
  await search(false);
}

watch(
  () => selected_experiment.value,
  () => { search() },
  { immediate: true }
);

</script>

<style lang="sass">
.q-table__bottom.row 
  justify-content: start

.q-table__separator
  flex: 0 0 0

.sticky-header-table-datafiles
  /* height or max-height is important */
  height: calc(100vh - 192px)
  width: calc(100vw - 243px)

  .q-table__top,
  .q-table__bottom,
  thead tr:first-child th
    /* bg color is important for th; just specify one */
    background-color: white

  thead tr th
    position: sticky
    z-index: 1
  thead tr:first-child th
    top: 0
    z-index: 2

  /* this is when the loading indicator appears */
  &.q-table--loading thead tr:last-child th
    /* height of all previous header rows */
    top: 48px

  /* prevent scrolling behind sticky top row on focus */
  tbody
    /* height of all previous header rows */
    scroll-margin-top: 48px

</style>
