<template>
  <div class="d-flex justify-content-between align-items-center mt-2">
    <div class="form-inline">
      <label class="form-label me-2">Rows per page:</label>
      <select class="form-select d-inline-block w-auto"
              :value="rowsPerPage"
              @change="onRowsPerPageChange($event)">
        <option :value="5">5</option>
        <option :value="10">10</option>
        <option :value="25">25</option>
        <option :value="50">50</option>
      </select>
    </div>
    <nav aria-label="Page navigation">
      <ul class="pagination mb-0">
        <li class="page-item" :class="{ disabled: page <= 1 }">
          <button class="page-link"
                  @click="emitGoTo(page - 1)"
                  :disabled="page <= 1">Previous</button>
        </li>
        <li class="page-item disabled"><span class="page-link">{{ page }} / {{ totalPages }}</span></li>
        <li class="page-item" :class="{ disabled: page >= totalPages }">
          <button class="page-link"
                  @click="emitGoTo(page + 1)"
                  :disabled="page >= totalPages">Next</button>
        </li>
      </ul>
    </nav>
  </div>
</template>

<script setup lang="ts">
import { computed, defineEmits, defineProps } from 'vue';

const props = defineProps({
  rowsPerPage: { type: Number, required: true },
  page: { type: Number, required: true },
  rowsNumber: { type: Number, required: true },
});

const emit = defineEmits(['updateRowsPerPage', 'goToPage']);

const totalPages = computed(() => {
  return props.rowsNumber > 0 ? Math.ceil(props.rowsNumber / props.rowsPerPage) : 1;
});

function onRowsPerPageChange(event: Event) {
  const target = event.target as HTMLSelectElement;
  const newVal = Number(target.value);
  emit('updateRowsPerPage', newVal);
}

function emitGoTo(newPage: number) {
  // Clamp within bounds
  const tp = totalPages.value;
  const safePage = Math.max(1, Math.min(newPage, tp));
  emit('goToPage', safePage);
}
</script>

<style scoped>
/* No custom styles needed – uses Bootstrap classes */
</style>
