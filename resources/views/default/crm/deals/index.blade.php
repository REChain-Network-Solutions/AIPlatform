@extends('panel.layout.app', ['disable_tblr' => true])

@section('title', __('Pipeline — CRM'))
@section('titlebar_title', __('Deal Pipeline'))
@section('titlebar_subtitle', $activePipeline?->name ?? __('No pipeline yet'))

@section('titlebar_actions')
    {{-- Pipeline switcher --}}
    @if ($pipelines->count() > 1)
        <div class="flex items-center gap-2">
            @foreach ($pipelines as $pipeline)
                <a
                    href="{{ route('dashboard.crm.deals.index', ['pipeline' => $pipeline->id]) }}"
                    @class(['lqd-btn lqd-btn-sm', 'lqd-btn-primary' => $activePipeline?->id === $pipeline->id, 'lqd-btn-outline' => $activePipeline?->id !== $pipeline->id])
                >{{ $pipeline->name }}</a>
            @endforeach
        </div>
    @endif
    <x-button
        href="{{ route('dashboard.crm.deals.create') }}"
        size="sm"
        variant="primary"
    >
        <x-tabler-plus class="w-4" />
        {{ __('New Deal') }}
    </x-button>
@endsection

@section('content')
    <div class="py-10">

        @if (! $activePipeline)
            <div class="py-20 text-center text-muted-foreground">
                <x-tabler-layout-kanban class="mx-auto mb-4 w-12 opacity-30" />
                <p class="mb-3">{{ __('No pipeline found. Create your first pipeline to get started.') }}</p>
            </div>
        @else
            {{-- Pipeline value bar --}}
            <div class="mb-6 flex items-center gap-6 text-sm">
                @php
                    $totalPipelineValue = $activePipeline->stages->flatMap->deals->where('status', 'open')->sum('value');
                @endphp
                <div>
                    <span class="text-muted-foreground">{{ __('Total Pipeline Value') }}:</span>
                    <strong class="ms-1 text-heading-foreground">
                        {{ $activePipeline->currency }} {{ number_format($totalPipelineValue, 0) }}
                    </strong>
                </div>
                <div>
                    <span class="text-muted-foreground">{{ __('Deals') }}:</span>
                    <strong class="ms-1 text-heading-foreground">
                        {{ $activePipeline->stages->flatMap->deals->where('status', 'open')->count() }}
                    </strong>
                </div>
            </div>

            {{-- Kanban Board --}}
            <div
                class="flex gap-4 overflow-x-auto pb-4"
                id="crm-kanban"
            >
                @foreach ($activePipeline->stages as $stage)
                    @php
                        $stageDeals = $stage->deals->where('status', 'open')->sortBy('order');
                        $stageValue = $stageDeals->sum('value');
                    @endphp
                    <div
                        class="flex w-72 shrink-0 flex-col rounded-2xl border border-border bg-muted/40"
                        data-stage-id="{{ $stage->id }}"
                    >
                        {{-- Stage header --}}
                        <div class="flex items-center justify-between rounded-t-2xl px-4 py-3">
                            <div class="flex items-center gap-2">
                                <span
                                    class="size-2.5 rounded-full"
                                    style="background: {{ $stage->color }}"
                                ></span>
                                <span class="text-sm font-semibold text-heading-foreground">{{ $stage->name }}</span>
                                <span class="rounded-full bg-muted px-1.5 py-0.5 text-xs text-muted-foreground">{{ $stageDeals->count() }}</span>
                            </div>
                            <span class="text-xs font-medium text-muted-foreground">
                                {{ $activePipeline->currency }} {{ number_format($stageValue, 0) }}
                            </span>
                        </div>

                        {{-- Deal cards --}}
                        <div
                            class="crm-stage-list min-h-[120px] flex-1 space-y-3 p-3"
                            data-stage-id="{{ $stage->id }}"
                        >
                            @foreach ($stageDeals as $deal)
                                <div
                                    class="crm-deal-card group cursor-grab rounded-xl border border-border bg-card p-4 shadow-xs transition hover:shadow-md active:cursor-grabbing"
                                    data-deal-id="{{ $deal->id }}"
                                >
                                    <a
                                        href="{{ route('dashboard.crm.deals.show', $deal) }}"
                                        class="mb-2 block font-medium text-heading-foreground hover:text-primary"
                                    >{{ $deal->title }}</a>

                                    @if ($deal->contact)
                                        <div class="mb-2 flex items-center gap-1.5 text-xs text-muted-foreground">
                                            <x-tabler-user class="w-3.5" />
                                            {{ $deal->contact->full_name }}
                                        </div>
                                    @endif

                                    <div class="flex items-center justify-between">
                                        <span class="text-sm font-semibold text-heading-foreground">
                                            {{ $deal->currency }} {{ number_format($deal->value, 0) }}
                                        </span>
                                        @if ($deal->expected_close_date)
                                            <span @class(['text-xs', 'text-red-500' => $deal->expected_close_date->isPast(), 'text-muted-foreground' => !$deal->expected_close_date->isPast()])>
                                                {{ $deal->expected_close_date->format('M d') }}
                                            </span>
                                        @endif
                                    </div>

                                    @if ($deal->probability !== null)
                                        <div class="mt-2">
                                            <div class="h-1.5 overflow-hidden rounded-full bg-muted">
                                                <div
                                                    class="h-full rounded-full bg-primary"
                                                    style="width: {{ $deal->probability }}%"
                                                ></div>
                                            </div>
                                            <p class="mt-0.5 text-right text-xs text-muted-foreground">{{ $deal->probability }}%</p>
                                        </div>
                                    @endif
                                </div>
                            @endforeach
                        </div>

                        {{-- Add deal quick link --}}
                        <a
                            href="{{ route('dashboard.crm.deals.create', ['stage' => $stage->id, 'pipeline' => $activePipeline->id]) }}"
                            class="flex items-center gap-1.5 rounded-b-2xl px-4 py-3 text-xs text-muted-foreground hover:bg-accent/50 hover:text-primary"
                        >
                            <x-tabler-plus class="w-3.5" /> {{ __('Add deal') }}
                        </a>
                    </div>
                @endforeach
            </div>

        @endif
    </div>
@endsection

@push('scripts')
    <script>
        // Lightweight kanban drag-drop using SortableJS (already a dependency)
        document.addEventListener('DOMContentLoaded', function () {
            const lists = document.querySelectorAll('.crm-stage-list');
            if (typeof Sortable === 'undefined') return;

            lists.forEach(function (list) {
                Sortable.create(list, {
                    group: 'crm-deals',
                    animation: 150,
                    ghostClass: 'opacity-40',
                    onEnd: function (evt) {
                        const dealId  = evt.item.dataset.dealId;
                        const stageId = evt.to.dataset.stageId;
                        const order   = evt.newIndex;

                        fetch(`/dashboard/crm/deals/${dealId}/move-stage`, {
                            method: 'PATCH',
                            headers: {
                                'Content-Type': 'application/json',
                                'X-CSRF-TOKEN': document.querySelector('meta[name="csrf-token"]').content,
                            },
                            body: JSON.stringify({ stage_id: stageId, order }),
                        });
                    },
                });
            });
        });
    </script>
@endpush
