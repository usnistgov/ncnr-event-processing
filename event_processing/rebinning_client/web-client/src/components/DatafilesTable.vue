<template>
  <div class="q-px-md">
    <div class="row flex">
      <div class="col-auto">
        <q-input
          v-model="search_inputs.filename"
          debounce="500"
          label="Filename Search"
          clearable
          @update:model-value="search"
        ></q-input>
      </div>
      <div class="col flex-1">
        <q-table
          class="my-sticky-column-table my-sticky-header-table"
          flat bordered
          :rows="rows"
          :columns="columns"
          row-key="filename"
          selection="single"
          v-model:selected="selected"
          @selection="on_selection"
          @row-dblclick="row_dblclick"
          wrap-cells
          v-model:pagination="pagination"
          @request="pagination_request_handler"
        />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import { api_get, ncnr_metadata_api, selected_experiment, selected_filename, selected_path, get_metadata } from 'src/store';

const endpoint = 'datafiles';
const columns = [
  { 'name': 'filename', 'label': 'Filename', 'field': 'filename', 'required': true },
  { 'name': 'cycle', 'label': 'Rx Cycle', 'field': 'rxcycle_id' },
  { 'name': 'start_date', 'label': 'Start Date', 'field': 'start_date' },
]


interface APISearchParams {
  offset: number,
  limit: number,
  experiment_id: string,
  filename?: string,
  full_count?: boolean
}

const rows = ref([]);
const selected = ref([]);
const pagination = ref({
  'rowsPerPage': 10,
  'descending': true,
  'sortBy': 'date',
  'page': 0,
  'rowsNumber': 0,
})

const search_inputs = ref({
  filename: '',
});

function on_selection() {
  const r = rows.value;
  if (r.length > 0) {
    console.log(r[0]);
    const { filename, localdir } = r[0];
    selected_filename.value = filename;
    selected_path.value = localdir;

    get_metadata();
  }
  else {
    selected_filename.value = selected_path.value = ''
  }
}

function row_dblclick(_evt: unknown, row: Row) {
  selected.value.splice(0, 1, row);
  on_selection();
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
  if (update_total) {
    const full_count_params: APISearchParams = {'full_count': true, ...params};
    const full_count_result = await api_get(ncnr_metadata_api, endpoint, full_count_params);
    if (full_count_result.length > 0) {
      const rowsNumber = full_count_result[0]?.full_count ?? 0;
      pagination.value['rowsNumber'] = rowsNumber;
      pagination.value['page'] = 0;
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

</script>

<style lang="sass">
.my-sticky-header-table
  /* height or max-height is important */
  max-height: 100%

  .q-table__top,
  .q-table__bottom,
  thead tr:first-child th
    /* bg color is important for th; just specify one */
    background-color: white

  thead tr th
    position: sticky
    z-index: 2
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

.my-sticky-column-table
  /* specifying max-width so the example can
    highlight the sticky column on any browser window */

  /* thead tr:first-child th:first-child
     bg color is important for th; just specify one
     background-color: white */

  td:first-child
    background-color: white

  th:first-child,
  td:first-child
    position: sticky
    left: 0
    z-index: 1
</style>
