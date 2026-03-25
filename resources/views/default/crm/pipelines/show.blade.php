@extends('panel.layout.app', ['disable_tblr' => true])

@section('title', $pipeline->name . ' — CRM Pipeline')
@section('titlebar_title', $pipeline->name)
@section('titlebar_subtitle', __('Pipeline stages and deals'))
@section('titlebar_back', route('dashboard.crm.pipelines.index'))

@section('titlebar_actions')
    <a
        href="{{ route('dashboard.crm.pipelines.edit', $pipeline) }}"
        class="lqd-btn lqd-btn-sm lqd-btn-outline"
    >
        <x-tabler-pencil class="w-4" /> {{ __('Edit Pipeline') }}
    </a>
    <a
        href="{{ route('dashboard.crm.deals.create') }}"
        class="lqd-btn lqd-btn-sm lqd-btn-primary"
    >
        <x-tabler-plus class="w-4" /> {{ __('New Deal') }}
    </a>
@endsection

@section('content')
    <div class="py-10">

        {{-- Pipeline meta --}}
        <div class="mb-6 flex items-center gap-4">
            @if ($pipeline->is_default)
                <span class="rounded-full bg-primary/10 px-3 py-1 text-xs font-medium text-primary">{{ __('Default') }}</span>
            @endif
            <span class="text-sm text-muted-foreground">{{ $pipeline->currency }}</span>
            <span class="text-sm text-muted-foreground">{{ $pipeline->stages->count() }} {{ __('stages') }}</span>
        </div>

        {{-- Kanban board --}}
        @if ($pipeline->stages->isEmpty())
            <div class="rounded-2xl border border-border bg-card py-20 text-center text-muted-foreground">
                <x-tabler-layout-kanban class="mx-auto mb-4 w-12 opacity-30" />
                <p class="mb-3">{{ __('No stages yet. Edit the pipeline to add stages.') }}</p>
                <a
                    href="{{ route('dashboard.crm.pipelines.edit', $pipeline) }}"
                    class="lqd-btn lqd-btn-sm lqd-btn-outline"
                >{{ __('Edit Pipeline') }}</a>
            </div>
        @else
            <div class="flex gap-4 overflow-x-auto pb-4">
                @foreach ($pipeline->stages as $stage)
                    @php
                        $stageDeals = $stage->deals->where('status', 'open');
                        $stageValue = $stageDeals->sum('value');
                    @endphp
                    <div class="w-72 shrink-0">
                        <div
                            class="mb-3 flex items-center justify-between rounded-xl px-3 py-2"
                            style="background-color: {{ $stage->color }}22; border-left: 3px solid {{ $stage->color }}"
                        >
                            <span class="font-semibold text-heading-foreground">{{ $stage->name }}</span>
                            <div class="flex items-center gap-2 text-xs text-muted-foreground">
                                @if ($stage->is_won)
                                    <span class="rounded-full bg-green-100 px-2 py-0.5 text-green-700">{{ __('Won') }}</span>
                                @elseif ($stage->is_lost)
                                    <span class="rounded-full bg-red-100 px-2 py-0.5 text-red-700">{{ __('Lost') }}</span>
                                @else
                                    <span>{{ $stage->probability }}%</span>
                                @endif
                                <span>{{ $stageDeals->count() }}</span>
                            </div>
                        </div>

                        @if ($stageValue > 0)
                            <p class="mb-2 px-1 text-xs text-muted-foreground">
                                {{ $pipeline->currency }} {{ number_format($stageValue, 2) }}
                            </p>
                        @endif

                        <div class="space-y-2">
                            @forelse ($stageDeals as $deal)
                                <a
                                    href="{{ route('dashboard.crm.deals.show', $deal) }}"
                                    class="block rounded-xl border border-border bg-card p-3 shadow-xs transition hover:border-primary/30 hover:shadow-sm"
                                >
                                    <p class="mb-1 font-medium text-heading-foreground">{{ $deal->title }}</p>
                                    @if ($deal->contact)
                                        <p class="text-xs text-muted-foreground">{{ $deal->contact->full_name }}</p>
                                    @endif
                                    @if ($deal->value)
                                        <p class="mt-2 text-sm font-semibold text-primary">
                                            {{ $pipeline->currency }} {{ number_format($deal->value, 2) }}
                                        </p>
                                    @endif
                                    @if ($deal->expected_close_date)
                                        <p class="mt-1 text-xs text-muted-foreground">
                                            <x-tabler-calendar class="inline w-3" />
                                            {{ $deal->expected_close_date->format('M j, Y') }}
                                        </p>
                                    @endif
                                </a>
                            @empty
                                <div class="rounded-xl border border-dashed border-border p-4 text-center text-xs text-muted-foreground">
                                    {{ __('No deals') }}
                                </div>
                            @endforelse
                        </div>
                    </div>
                @endforeach
            </div>
        @endif

    </div>
@endsection
