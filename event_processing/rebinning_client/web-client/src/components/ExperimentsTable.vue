<template>
  <div class="q-px-md items-stretch">
    <div class="row flex items-stretch">
      <div class="col-auto">
        <q-select
          v-model="search_inputs.instrument_name"
          :options="instrument_names"
          label="Instrument Name"
          @update:model-value="search"
          clearable
          ></q-select>
        <q-input
          v-model="search_inputs.participant_name"
          debounce="500"
          label="Participant Name"
          clearable
          @update:model-value="search"
        ></q-input>
      </div>
      <div class="col flex-1">
        <q-table
          class="my-sticky-column-table my-sticky-header-table"
          flat bordered dense
          :rows="rows"
          :columns="columns"
          row-key="id"
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
import { api_get, ncnr_metadata_api, all_instruments, selected_experiment } from 'src/store';

const endpoint = 'experiments';
const columns = [
    {'name': 'experiment_id', 'label': 'Experiment ID', 'field': 'id', 'required': true, 'align': 'left', 'style': 'width: 8em;'},
    {'name': 'title', 'label': 'Title', 'field': 'title', 'align': 'left', 'style': 'max-width:2px;'},
    {'name': 'participants', 'label': 'Participants', 'field': 'participant_names', 'align': 'left', 'format': (value: string) => JSON.parse(value).join(', '), 'style': 'max-width:2px;'},
]

interface Row {
  id: string,
  title: string,
  participant_names: string[],
}

interface APISearchParams {
  offset: number,
  limit: number,
  instrument_id?: string,
  participant_name?: string,
  title?: string,
  id?: string,
  full_count?: boolean
}

const rows = ref<Row[]>([]);
const selected = ref<Row[]>([]);
const pagination = ref({
  'rowsPerPage': 10,
  'descending': true,
  'sortBy': 'date',
  'page': 0,
  'rowsNumber': 0,
})

const search_inputs = ref({
  instrument_name: '',
  experiment_title: '',
  experiment_id: '',
  participant_name: '',
});

function on_selection(event: { keys: string[] }) {
  const { keys } = event;
  if (keys.length > 0) {
    selected_experiment.value = keys[0];
  }
}

function row_dblclick(_evt: unknown, row: Row) {
  selected.value.splice(0, 1, row);
  selected_experiment.value = row.id;
}

// await (await fetch(`${ncnr_metadata_api}/instruments`)).json();
const instrument_names = [
    'vsans',
    'macs',
    'candor',
]

async function search(update_total = true) {
  // update_total is True for new searches, but not when paginating
  // reset offset for new searches:
  const { rowsPerPage, page } = pagination.value;
  const offset = (update_total) ? 0 : rowsPerPage * (page - 1);
  const params: APISearchParams = { offset, limit: pagination.value.rowsPerPage }
  const { instrument_name, participant_name, experiment_id, experiment_title } = search_inputs.value;
  if (instrument_name) {
    const instrument_id = all_instruments.value.find((instr) => (instr.alias === instrument_name))?.id;
    params['instrument_id'] = instrument_id;
  }
  if (participant_name) {
    params['participant_name'] = `%${participant_name}%`;
  }
  if (experiment_title) {
    params['title'] = `%${experiment_title}%`;
  }
  if (experiment_id) {
    params['id'] = `${experiment_id}`;
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
