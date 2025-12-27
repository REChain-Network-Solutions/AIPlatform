@php
    //Replicating table styles from table component
    $base_class = 'transition-colors max-lg:group-[&[data-view-mode=list]]:overflow-x-auto';

    $variations = [
        'variant' => [
            'solid' => 'rounded-card bg-card-background pt-1 group-[&[data-view-mode=grid]]:bg-transparent',
            'outline' => 'rounded-card border border-card-border pt-1 group-[&[data-view-mode=grid]]:border-0',
            'shadow' => ' rounded-card shadow-card bg-card-background pt-1 group-[&[data-view-mode=grid]]:shadow-none group-[&[data-view-mode=grid]]:bg-transparent',
            'outline-shadow' => 'rounded-card border border-card-border pt-1 shadow-card bg-card-background',
            'plain' => '',
        ],
    ];

    $variant =
        isset($variant) && isset($variations['variant'][$variant])
            ? $variations['variant'][$variant]
            : $variations['variant'][Theme::getSetting('defaultVariations.table.variant', 'outline')];

    $class = @twMerge($base_class, $variant);
@endphp

<div
    class="lqd-posts-container lqd-docs-container group transition-all [&[aria-busy=true]]:animate-pulse max-lg:[&[data-view-mode=list]]:max-w-full"
    id="lqd-docs-container"
    data-view-mode="list"
    x-bind:data-view-mode="$store.docsViewMode.docsViewMode"
    x-init="$store.documentsSelection.clearSelection()"
    x-merge.transition
>
    {{-- Setting the view mode attribute before contents load to avoid page flashes --}}
    <script>
        document.querySelector('.lqd-docs-container')?.setAttribute('data-view-mode', localStorage.getItem('docsViewMode')?.replace(/\"/g, '') || 'list');
    </script>

    <div class="{{ $class }}">
        <div
            class="lqd-posts-head lqd-docs-head grid items-center gap-x-4 border-b px-4 py-3 text-4xs font-medium uppercase leading-tight tracking-wider text-foreground/50 [grid-template-columns:3fr_repeat(2,minmax(0,1fr))_100px_1fr] group-[&[data-view-mode=grid]]:hidden">
            <span class="inline-flex items-center gap-3">
                @if (request()->route()->getName() !== 'dashboard.user.index')
                    <label
                        class="document-checkbox-label relative z-10 inline-grid size-[18px] cursor-pointer select-none place-items-center rounded bg-foreground/5 text-primary before:absolute before:left-1/2 before:top-1/2 before:size-8 before:-translate-x-1/2 before:-translate-y-1/2"
                        for="doc-select-all-visible"
                    >
                        <input
                            class="document-checkbox-all-visible peer invisible absolute z-10 size-0"
                            id="doc-select-all-visible"
                            data-id="doc-select-all-visible"
                            type="checkbox"
                            x-init=""
                            @change.default="Alpine.store('documentsSelection').updateSelectedItems({ checkboxEl: $el })"
                        />
                        <span class="col-start-1 col-end-1 row-start-1 row-end-1 inline-block size-full rounded bg-primary/5 opacity-0 transition peer-checked:opacity-100"></span>
                        <x-tabler-check
                            class="col-start-1 col-end-1 row-start-1 row-end-1 size-4 scale-75 opacity-0 transition peer-checked:scale-100 peer-checked:opacity-100"
                            stroke-width="2.5"
                        />
                        <x-tabler-minus
                            class="col-start-1 col-end-1 row-start-1 row-end-1 size-4 scale-75 opacity-0 transition peer-checked:scale-100 peer-[&.partial]:opacity-100"
                            stroke-width="3"
                        />
                    </label>
                @endif
                {{ __('Name') }}
            </span>

            <span>
                {{ __('Type') }}
            </span>

            <span>
                {{ __('Date') }}
            </span>

            <span>
                {{ __('Cost') }}
            </span>

            <span class="text-center">
                {{ __('Actions') }}
            </span>
        </div>

        @include('panel.user.openai.documents_list')
    </div>

    @if (!isset($disablePagination))
        {{ $items->links('pagination::ajax', [
            'action' => route('dashboard.user.openai.documents.all', ['id' => $currfolder?->id, 'listOnly' => true]),
            'currfolder' => $currfolder,
            'target_id' => 'lqd-docs-container',
        ]) }}
    @endif
</div>

<div
    class="pointer-events-none fixed bottom-8 end-0 start-0 z-10 transition-all lg:start-[--navbar-width]"
    x-init=""
    x-cloak
    x-show="$store.documentsSelection.selectedItems.length > 0"
    x-transition.scale-95
>
    <div class="container">
        <form
            class="pointer-events-auto flex flex-col items-center justify-between gap-1 rounded-full border border-foreground/5 bg-background px-6 py-4 shadow-xl shadow-black/5 md:flex-row md:py-1 md:pe-1"
            x-data="{ selectedAction: 'delete' }"
            @submit.prevent="
                if (selectedAction === 'delete') {
                    $store.documentsSelection.bulkDelete('selected', {
                        confirmSelectedMessage: '{{ __('Are you sure you want to delete the selected documents?') }}',
                        noSelectionMessage: '{{ __('Please select documents to delete.') }}',
                        deleteUrl: '{{ route('dashboard.user.openai.documents.bulkDelete') }}'
                    });
                }
            "
        >
            <p class="m-0 text-2xs font-medium">
                <span x-text="$store.documentsSelection.selectedItems.length"></span>
                <span
                    x-show="$store.documentsSelection.selectedItems.length === 1"
                    x-text="'{{ __('Item') }}'"
                ></span>
                <span
                    x-show="$store.documentsSelection.selectedItems.length !== 1"
                    x-text="'{{ __('Items') }}'"
                ></span>
                {{ __('selected') }}
            </p>
            <div class="flex items-center gap-3">
                <x-forms.input
                    class="w-full rounded-full md:w-auto md:pe-12"
                    type="select"
                    size="md"
                    x-model="selectedAction"
                >
                    <option value="delete">
                        {{ __('Move to Trash') }}
                    </option>
                </x-forms.input>

                <x-button
                    type="submit"
                    size="md"
                >
                    {{ __('Apply') }}
                </x-button>
            </div>
        </form>
    </div>
</div>
