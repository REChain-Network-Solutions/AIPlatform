<script>
(function () {
    let stageIndex = document.querySelectorAll('.stage-row').length;

    document.getElementById('add-stage')?.addEventListener('click', () => {
        const list = document.getElementById('stages-list');
        const i = stageIndex++;
        const row = document.createElement('div');
        row.className = 'stage-row flex items-center gap-3 rounded-xl border border-border bg-background p-3';
        row.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" class="w-4 shrink-0 cursor-grab text-muted-foreground/50" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="9" cy="5" r="1"/><circle cx="9" cy="12" r="1"/><circle cx="9" cy="19" r="1"/><circle cx="15" cy="5" r="1"/><circle cx="15" cy="12" r="1"/><circle cx="15" cy="19" r="1"/></svg>
            <input type="color" name="stages[${i}][color]" value="#6366f1" class="h-8 w-8 shrink-0 cursor-pointer rounded-lg border-0 p-0.5">
            <input type="text" name="stages[${i}][name]" placeholder="{{ __('Stage name') }}" required class="lqd-input lqd-input-sm flex-1 min-w-0">
            <div class="flex items-center gap-1 text-xs text-muted-foreground shrink-0">
                <input type="number" name="stages[${i}][probability]" value="0" min="0" max="100" class="lqd-input lqd-input-sm w-14 text-center"> %
            </div>
            <label class="flex items-center gap-1 text-xs text-muted-foreground shrink-0 cursor-pointer">
                <input type="checkbox" name="stages[${i}][is_won]" value="1" class="lqd-checkbox"> {{ __('Won') }}
            </label>
            <label class="flex items-center gap-1 text-xs text-muted-foreground shrink-0 cursor-pointer">
                <input type="checkbox" name="stages[${i}][is_lost]" value="1" class="lqd-checkbox"> {{ __('Lost') }}
            </label>
            <button type="button" class="remove-stage shrink-0 text-muted-foreground hover:text-red-500">
                <svg xmlns="http://www.w3.org/2000/svg" class="w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12"/></svg>
            </button>
        `;
        list.appendChild(row);
        bindRemove(row.querySelector('.remove-stage'));
        row.querySelector('input[type="text"]').focus();
    });

    function bindRemove(btn) {
        btn?.addEventListener('click', () => btn.closest('.stage-row').remove());
    }

    document.querySelectorAll('.remove-stage').forEach(bindRemove);
})();
</script>
