@extends('panel.layout.app', ['disable_tblr' => true])
@section('title', __('Shared Credit Costs'))
@section('content')
    <div
        class="py-10"
        x-data="creditCostManager"
    >
        {{-- Cost Calculation Notice --}}
        <x-alert
            variant="info-fill"
            size="lg"
        >
            <p class="font-semibold">{{ __('How Credits Are Calculated') }}</p>
            <p class="inline-block rounded bg-black/5 px-2 py-1 font-mono text-xs dark:bg-white/10">{{ __('Credits Deducted = Units × Cost Index') }}</p>
            <ul class="list-disc ps-4 text-xs">
                <li><strong>{{ __('Text / Chat') }}:</strong> {{ __('per word generated (output only)') }}</li>
                <li><strong>{{ __('Text-to-Speech') }}:</strong> {{ __('per character synthesized') }}</li>
                <li><strong>{{ __('Image Generation') }}:</strong> {{ __('per image generated') }}</li>
                <li><strong>{{ __('Video Generation') }}:</strong> {{ __('per second / minute of video') }}</li>
                <li><strong>{{ __('Speech-to-Text') }}:</strong> {{ __('per minute transcribed') }}</li>
                <li><strong>{{ __('Embeddings') }}:</strong> {{ __('per word embedded') }}</li>
            </ul>
            <p class="text-xs opacity-70 mt-4">{{ __('Overrides replace the Default Index when active. Input tokens are not separately charged — only output is metered.') }}</p>
        </x-alert>

        {{-- Filters --}}
        <div class="my-6 flex flex-wrap items-center gap-3">
            <div class="relative grow sm:max-w-xs">
                <x-tabler-search class="size-4 absolute start-3 top-1/2 -translate-y-1/2 opacity-40 pointer-events-none" />
                <input
                    class="lqd-input w-full ps-10 pe-4 py-2 border border-input-border bg-input-background text-input-foreground rounded-input text-2xs transition-colors focus:border-secondary focus:outline-0 focus:ring focus:ring-secondary h-10"
                    type="text"
                    placeholder="{{ __('Search models...') }}"
                    x-model="search"
                >
            </div>
            <x-forms.input
                class:container="w-auto"
                class="h-10"
                name="engine_filter"
                type="select"
                size="none"
                x-model="engineFilter"
            >
                <option value="">{{ __('All Engines') }}</option>
                @foreach ($entities->map(fn($e) => $e->engine()->value)->unique()->sort() as $engine)
                    <option value="{{ $engine }}">{{ $engine }}</option>
                @endforeach
            </x-forms.input>
            <x-forms.input
                class:container="flex items-center"
                id="overrides_only_filter"
                name="overrides_only"
                type="checkbox"
                switcher
                value="1"
                label="{{ __('Overrides only') }}"
                x-model="showOverridesOnly"
            />
            <span
                class="ms-auto text-2xs opacity-60"
                x-text="'{{ __('Showing') }} ' + filteredCount + ' {{ __('of') }} {{ $entities->count() }} {{ __('models') }}'"
            ></span>
        </div>

        {{-- Table --}}
        <x-card
            class="overflow-hidden"
            size="none"
        >
            <x-table class:table="table-sm">
                <x-slot:head>
                    <tr>
                        <th class="ps-6">{{ __('Model') }}</th>
                        <th>{{ __('Engine') }}</th>
                        <th class="text-end">{{ __('Default Index') }}</th>
                        <th class="text-end">{{ __('Active Cost') }}</th>
                        <th class="text-center">{{ __('Status') }}</th>
                        <th class="text-end pe-6">{{ __('Actions') }}</th>
                    </tr>
                </x-slot:head>

                <x-slot:body>
                    @foreach ($entities as $entity)
                        @php
                            $slug = $entity->slug();
                            $override = $overrides[$slug] ?? null;
                            $engineValue = $entity->engine()->value;
                            $label = $entity->label();
                            $defaultCost = $entity->sharedCreditIndex();
                            $defaultCostDecimals = $defaultCost < 1 ? 4 : 2;
                        @endphp
                        <tr
                            x-show="matchesFilter('{{ $slug }}', '{{ $label }}', '{{ $engineValue }}', {{ $override ? 'true' : 'false' }})"
                            x-transition.opacity
                        >
                            <td class="ps-6">
                                <div class="flex items-center gap-2">
                                    <div>
                                        <span class="font-medium text-heading-foreground">{{ $label }}</span>
                                        <span class="block text-2xs opacity-50 font-mono">{{ $slug }}</span>
                                    </div>
                                </div>
                            </td>
                            <td>
                                <x-badge
                                    class="text-2xs"
                                    variant="secondary"
                                >
                                    {{ $engineValue }}
                                </x-badge>
                            </td>
                            <td class="text-end font-mono text-2xs opacity-60">
                                {{ number_format($defaultCost, $defaultCostDecimals) }}
                            </td>
                            <td class="text-end">
                                @if ($override)
                                    <span class="font-mono font-semibold text-primary">
                                        {{ number_format($override->base_cost, $defaultCostDecimals) }}
                                    </span>
                                @else
                                    <span class="font-mono">{{ number_format($defaultCost, $defaultCostDecimals) }}</span>
                                @endif
                            </td>
                            <td class="text-center">
                                @if ($override)
                                    <x-badge
                                        class="text-2xs"
                                        variant="{{ $override->is_active ? 'success' : 'danger' }}"
                                    >
                                        {{ $override->is_active ? __('Override') : __('Disabled') }}
                                    </x-badge>
                                @else
                                    <span class="text-2xs opacity-40">{{ __('Default') }}</span>
                                @endif
                            </td>
                            <td class="whitespace-nowrap text-end pe-6">
                                @if ($override)
                                    {{-- Edit Override --}}
                                    <x-modal
                                        class="inline-flex"
                                        title="{{ __('Edit Cost Override') }} - {{ $label }}"
                                    >
                                        <x-slot:trigger
                                            class="size-9"
                                            size="none"
                                            variant="ghost-shadow"
                                            title="{{ __('Edit Override') }}"
                                        >
                                            <x-tabler-pencil class="size-4" />
                                        </x-slot:trigger>

                                        <x-slot:modal>
                                            <form
                                                class="flex flex-wrap gap-y-5"
                                                method="POST"
                                                action="{{ route('dashboard.admin.finance.shared-credit-costs.update', $override) }}"
                                            >
                                                @csrf
                                                @method('PUT')

                                                <div class="w-full rounded-lg bg-surface p-4">
                                                    <div class="flex items-center justify-between text-2xs">
                                                        <span class="opacity-60">{{ __('Model') }}</span>
                                                        <span class="font-medium">{{ $label }}</span>
                                                    </div>
                                                    <div class="mt-2 flex items-center justify-between text-2xs">
                                                        <span class="opacity-60">{{ __('Default Index') }}</span>
                                                        <span class="font-mono">{{ number_format($defaultCost, $defaultCostDecimals) }}</span>
                                                    </div>
                                                </div>

                                                <x-forms.input
                                                    class:container="w-full"
                                                    label="{{ __('Override Cost') }}"
                                                    name="base_cost"
                                                    type="number"
                                                    step="0.01"
                                                    min="0"
                                                    size="lg"
                                                    value="{{ $override->base_cost }}"
                                                    required
                                                />

                                                <x-forms.input
                                                    class:container="w-full"
                                                    id="is_active_edit_{{ $override->id }}"
                                                    name="is_active"
                                                    type="checkbox"
                                                    switcher
                                                    :checked="$override->is_active"
                                                    value="1"
                                                    label="{{ __('Active') }}"
                                                />

                                                <div class="mt-2 flex w-full gap-3 border-t border-card-border pt-5">
                                                    <x-button
                                                        @click.prevent="modalOpen = false"
                                                        variant="outline"
                                                        type="button"
                                                    >
                                                        {{ __('Cancel') }}
                                                    </x-button>
                                                    <x-button type="submit">
                                                        {{ __('Save Changes') }}
                                                    </x-button>
                                                </div>
                                            </form>
                                        </x-slot:modal>
                                    </x-modal>

                                    {{-- Delete Override --}}
                                    <x-modal
                                        class="inline-flex"
                                        title="{{ __('Remove Override') }}"
                                    >
                                        <x-slot:trigger
                                            class="size-9"
                                            size="none"
                                            variant="ghost-shadow"
                                            hover-variant="danger"
                                            title="{{ __('Remove Override') }}"
                                        >
                                            <x-tabler-trash class="size-4" />
                                        </x-slot:trigger>

                                        <x-slot:modal>
                                            <div class="flex flex-col items-center gap-4 text-center">
                                                <div class="flex size-14 items-center justify-center rounded-full bg-red-100 text-red-600 dark:bg-red-500/10">
                                                    <x-tabler-alert-triangle class="size-7" />
                                                </div>
                                                <p class="text-lg font-semibold">
                                                    {{ __('Remove override for :model?', ['model' => $label]) }}
                                                </p>
                                                <p class="text-sm opacity-70">
                                                    {{ __('This model will revert to its default cost of :cost credits.', ['cost' => number_format($defaultCost, 2)]) }}
                                                </p>
                                                <div class="mt-2 flex w-full gap-3 border-t border-card-border pt-5">
                                                    <x-button
                                                        class="w-full"
                                                        @click.prevent="modalOpen = false"
                                                        variant="outline"
                                                        type="button"
                                                    >
                                                        {{ __('Cancel') }}
                                                    </x-button>
                                                    <form
                                                        class="w-full"
                                                        method="POST"
                                                        action="{{ route('dashboard.admin.finance.shared-credit-costs.destroy', $override) }}"
                                                    >
                                                        @csrf
                                                        @method('DELETE')
                                                        <x-button
                                                            class="w-full"
                                                            variant="danger"
                                                            type="submit"
                                                        >
                                                            {{ __('Remove Override') }}
                                                        </x-button>
                                                    </form>
                                                </div>
                                            </div>
                                        </x-slot:modal>
                                    </x-modal>
                                @else
                                    {{-- Add Override --}}
                                    <x-modal
                                        class="inline-flex"
                                        title="{{ __('Add Cost Override') }} - {{ $label }}"
                                    >
                                        <x-slot:trigger
                                            class="size-9"
                                            size="none"
                                            variant="ghost-shadow"
                                            title="{{ __('Add Override') }}"
                                        >
                                            <x-tabler-plus class="size-4" />
                                        </x-slot:trigger>

                                        <x-slot:modal>
                                            <form
                                                class="flex flex-wrap gap-y-5"
                                                method="POST"
                                                action="{{ route('dashboard.admin.finance.shared-credit-costs.store') }}"
                                            >
                                                @csrf
                                                <input
                                                    type="hidden"
                                                    name="entity_key"
                                                    value="{{ $slug }}"
                                                >

                                                <div class="w-full rounded-lg bg-surface p-4">
                                                    <div class="flex items-center justify-between text-2xs">
                                                        <span class="opacity-60">{{ __('Model') }}</span>
                                                        <span class="font-medium">{{ $label }}</span>
                                                    </div>
                                                    <div class="mt-2 flex items-center justify-between text-2xs">
                                                        <span class="opacity-60">{{ __('Engine') }}</span>
                                                        <span>{{ $engineValue }}</span>
                                                    </div>
                                                    <div class="mt-2 flex items-center justify-between text-2xs">
                                                        <span class="opacity-60">{{ __('Current Default Cost') }}</span>
                                                        <span class="font-mono font-semibold">{{ number_format($defaultCost, 2) }}</span>
                                                    </div>
                                                </div>

                                                <x-forms.input
                                                    class:container="w-full"
                                                    label="{{ __('Override Cost') }}"
                                                    name="base_cost"
                                                    type="number"
                                                    step="0.01"
                                                    min="0"
                                                    size="lg"
                                                    value="{{ $defaultCost }}"
                                                    required
                                                />

                                                <x-forms.input
                                                    class:container="w-full"
                                                    id="is_active_{{ $slug }}"
                                                    name="is_active"
                                                    type="checkbox"
                                                    switcher
                                                    :checked="true"
                                                    value="1"
                                                    label="{{ __('Active') }}"
                                                />

                                                <div class="mt-2 flex w-full gap-3 border-t border-card-border pt-5">
                                                    <x-button
                                                        @click.prevent="modalOpen = false"
                                                        variant="outline"
                                                        type="button"
                                                    >
                                                        {{ __('Cancel') }}
                                                    </x-button>
                                                    <x-button type="submit">
                                                        {{ __('Save Override') }}
                                                    </x-button>
                                                </div>
                                            </form>
                                        </x-slot:modal>
                                    </x-modal>
                                @endif
                            </td>
                        </tr>
                    @endforeach
                </x-slot:body>
            </x-table>

            {{-- Empty filtered state --}}
            <div
                class="py-10 text-center"
                x-show="filteredCount === 0"
                x-cloak
            >
                <x-tabler-search-off class="mx-auto size-10 opacity-30" />
                <p class="mt-3 text-sm opacity-60">{{ __('No models match your filters.') }}</p>
                <x-button
                    class="mt-3"
                    variant="ghost-shadow"
                    size="sm"
                    @click="search = ''; engineFilter = ''; showOverridesOnly = false"
                >
                    {{ __('Clear Filters') }}
                </x-button>
            </div>
        </x-card>
    </div>
@endsection

@push('script')
    <script>
        document.addEventListener('alpine:init', () => {
            Alpine.data('creditCostManager', () => ({
                search: '',
                engineFilter: '',
                showOverridesOnly: false,
                totalCount: {{ $entities->count() }},
                entities: @json($entityData),

                get filteredCount() {
                    return this.entities.filter(e => this.isVisible(e)).length;
                },

                isVisible(entity) {
                    const searchLower = this.search.toLowerCase();
                    const matchesSearch = !searchLower ||
                        entity.slug.toLowerCase().includes(searchLower) ||
                        entity.label.toLowerCase().includes(searchLower);
                    const matchesEngine = !this.engineFilter || entity.engine === this.engineFilter;
                    const matchesOverride = !this.showOverridesOnly || entity.hasOverride;

                    return matchesSearch && matchesEngine && matchesOverride;
                },

                matchesFilter(slug, label, engine, hasOverride) {
                    return this.isVisible({ slug, label, engine, hasOverride });
                }
            }));
        });
    </script>
@endpush
