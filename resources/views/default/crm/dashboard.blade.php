@extends('panel.layout.app', ['disable_tblr' => true])

@section('title', __('CRM — Titan BOS'))

@section('titlebar_title', __('CRM Dashboard'))

@section('titlebar_subtitle', __('Your pipeline, contacts, and activities at a glance.'))

@section('titlebar_actions')
    <x-button
        href="{{ route('dashboard.crm.contacts.create') }}"
        size="sm"
        variant="primary"
    >
        <x-tabler-plus class="w-4" />
        {{ __('Add Contact') }}
    </x-button>
    <x-button
        href="{{ route('dashboard.crm.deals.create') }}"
        size="sm"
        variant="outline"
    >
        <x-tabler-currency-dollar class="w-4" />
        {{ __('New Deal') }}
    </x-button>
@endsection

@section('content')
    <div class="py-10">

        {{-- KPI Cards --}}
        <div class="mb-8 grid grid-cols-2 gap-4 lg:grid-cols-3 xl:grid-cols-6">
            @php
                $kpis = [
                    ['label' => __('Contacts'), 'value' => number_format($stats['contacts']), 'icon' => 'tabler-users', 'color' => 'text-blue-500', 'href' => route('dashboard.crm.contacts.index')],
                    ['label' => __('Companies'), 'value' => number_format($stats['companies']), 'icon' => 'tabler-building', 'color' => 'text-violet-500', 'href' => route('dashboard.crm.companies.index')],
                    ['label' => __('Open Deals'), 'value' => number_format($stats['open_deals']), 'icon' => 'tabler-target', 'color' => 'text-amber-500', 'href' => route('dashboard.crm.deals.index')],
                    ['label' => __('Pipeline Value'), 'value' => '$' . number_format($stats['pipeline_value'], 0), 'icon' => 'tabler-chart-bar', 'color' => 'text-green-500', 'href' => route('dashboard.crm.deals.index')],
                    ['label' => __('Won This Month'), 'value' => '$' . number_format($stats['won_this_month'], 0), 'icon' => 'tabler-trophy', 'color' => 'text-emerald-500', 'href' => route('dashboard.crm.deals.index')],
                    ['label' => __('Overdue Tasks'), 'value' => number_format($stats['overdue_tasks']), 'icon' => 'tabler-alarm', 'color' => 'text-red-500', 'href' => route('dashboard.crm.activities.index', ['overdue' => 1])],
                ];
            @endphp

            @foreach ($kpis as $kpi)
                <a
                    href="{{ $kpi['href'] }}"
                    class="flex flex-col gap-3 rounded-2xl border border-border bg-card p-5 shadow-xs transition hover:shadow-md"
                >
                    <div class="flex items-center justify-between">
                        <span class="text-xs font-medium text-muted-foreground">{{ $kpi['label'] }}</span>
                        <x-dynamic-component
                            :component="$kpi['icon']"
                            class="w-5 {{ $kpi['color'] }}"
                        />
                    </div>
                    <p class="text-2xl font-bold text-heading-foreground">{{ $kpi['value'] }}</p>
                </a>
            @endforeach
        </div>

        <div class="grid grid-cols-1 gap-6 lg:grid-cols-3">

            {{-- Recent Deals --}}
            <div class="col-span-2 rounded-2xl border border-border bg-card p-6 shadow-xs">
                <div class="mb-4 flex items-center justify-between">
                    <h2 class="text-sm font-semibold text-heading-foreground">{{ __('Top Open Deals') }}</h2>
                    <a
                        class="text-xs text-primary hover:underline"
                        href="{{ route('dashboard.crm.deals.index') }}"
                    >{{ __('View all') }}</a>
                </div>
                @forelse ($recentDeals as $deal)
                    <a
                        href="{{ route('dashboard.crm.deals.show', $deal) }}"
                        class="flex items-center justify-between border-b border-border py-3 last:border-0 hover:opacity-80"
                    >
                        <div>
                            <p class="font-medium text-heading-foreground">{{ $deal->title }}</p>
                            <p class="text-xs text-muted-foreground">
                                {{ $deal->contact?->full_name ?? '—' }}
                                @if ($deal->stage)
                                    · <span
                                        class="rounded px-1 py-0.5 text-xs"
                                        style="background: {{ $deal->stage->color }}22; color: {{ $deal->stage->color }}"
                                    >{{ $deal->stage->name }}</span>
                                @endif
                            </p>
                        </div>
                        <span class="font-semibold text-heading-foreground">
                            {{ $deal->currency }} {{ number_format($deal->value, 0) }}
                        </span>
                    </a>
                @empty
                    <x-empty-state>{{ __('No open deals yet.') }}</x-empty-state>
                @endforelse
            </div>

            {{-- Upcoming Activities --}}
            <div class="rounded-2xl border border-border bg-card p-6 shadow-xs">
                <div class="mb-4 flex items-center justify-between">
                    <h2 class="text-sm font-semibold text-heading-foreground">{{ __('Upcoming Activities') }}</h2>
                    <a
                        class="text-xs text-primary hover:underline"
                        href="{{ route('dashboard.crm.activities.index') }}"
                    >{{ __('View all') }}</a>
                </div>
                @forelse ($upcomingActivities as $activity)
                    <div class="flex items-start gap-3 border-b border-border py-3 last:border-0">
                        <div class="mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary">
                            <x-dynamic-component
                                :component="$activity->type_icon"
                                class="w-3.5"
                            />
                        </div>
                        <div class="min-w-0 flex-1">
                            <p class="truncate text-xs font-medium text-heading-foreground">{{ $activity->subject }}</p>
                            <p class="text-xs text-muted-foreground">
                                @if ($activity->due_at)
                                    {{ $activity->due_at->diffForHumans() }}
                                    @if ($activity->due_at->isPast())
                                        <span class="text-red-500">· {{ __('Overdue') }}</span>
                                    @endif
                                @else
                                    {{ __('No due date') }}
                                @endif
                            </p>
                        </div>
                    </div>
                @empty
                    <x-empty-state>{{ __('No upcoming activities.') }}</x-empty-state>
                @endforelse
            </div>

        </div>

        {{-- Recent Contacts --}}
        <div class="mt-6 rounded-2xl border border-border bg-card p-6 shadow-xs">
            <div class="mb-4 flex items-center justify-between">
                <h2 class="text-sm font-semibold text-heading-foreground">{{ __('Recent Contacts') }}</h2>
                <a
                    class="text-xs text-primary hover:underline"
                    href="{{ route('dashboard.crm.contacts.index') }}"
                >{{ __('View all') }}</a>
            </div>
            <div class="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-5">
                @forelse ($recentContacts as $contact)
                    <a
                        href="{{ route('dashboard.crm.contacts.show', $contact) }}"
                        class="flex items-center gap-3 rounded-xl border border-border p-3 hover:bg-accent/50 hover:shadow-xs"
                    >
                        <div class="flex size-9 shrink-0 items-center justify-center rounded-full bg-primary/10 text-sm font-bold text-primary">
                            {{ strtoupper(substr($contact->first_name, 0, 1)) }}{{ strtoupper(substr($contact->last_name ?? '', 0, 1)) }}
                        </div>
                        <div class="min-w-0">
                            <p class="truncate text-xs font-medium text-heading-foreground">{{ $contact->full_name }}</p>
                            <p class="truncate text-xs text-muted-foreground">{{ $contact->company_name ?? $contact->company?->name ?? '—' }}</p>
                        </div>
                    </a>
                @empty
                    <p class="text-xs text-muted-foreground">{{ __('No contacts yet.') }}</p>
                @endforelse
            </div>
        </div>

    </div>
@endsection
