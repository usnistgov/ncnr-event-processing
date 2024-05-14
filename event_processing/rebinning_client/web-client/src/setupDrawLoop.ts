import { ref, onActivated, onDeactivated, shallowRef } from 'vue';

export function setupDrawLoop<DrawArgsType>(draw: (args: DrawArgsType) => Promise<void>, name: string = '') {
  const mounted = ref(false);
  const drawing_busy = ref(false);
  const draw_requested = shallowRef<null | DrawArgsType>(null);

  const draw_if_needed = async function() {
    if (!mounted.value) {
      return;
    }
    if (drawing_busy.value) {
      console.log(`drawing ${name} busy!`);
    }
    else if (draw_requested.value !== null) {
      drawing_busy.value = true;
      const draw_args = draw_requested.value as DrawArgsType;
      draw_requested.value = null;
      try {
        // Need to continue the draw loop even if draw fails
        await draw(draw_args);
      }
      catch (e) {
        // TODO: should this notify the user?
        console.error(`Error drawing ${name} with args ${draw_args}:`, e);
        // add sleep to avoid runaway error loop
        await sleep(1000);
      }
      drawing_busy.value = false;
    }
    window.requestAnimationFrame(draw_if_needed);
  }

  const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

  onActivated(async () => {
    mounted.value = true;
    window.requestAnimationFrame(draw_if_needed);
  });

  onDeactivated(() => {
    mounted.value = false;
  });

  return { mounted, drawing_busy, draw_requested };
}
