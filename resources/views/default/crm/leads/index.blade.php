@extends('panel.layout.app', ['disable_tblr' => true])

@section('title', __('Leads — CRM'))
@section('titlebar_title', __('Leads'))
@section('titlebar_subtitle', __('Track and convert your incoming leads.'))

@section('titlebar_actions')
    {{-- View toggle --}}
    <div class="flex items-center gap-1 rounded-lg border border-border bg-muted p-0.5">
        <a
            href="{{ route('dashboard.crm.leads.index', array_merge(request()->query(), ['view' => 'kanban'])) }}"
            @class(['flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm transition', 'bg-background shadow-xs text-heading-foreground' => $view === 'kanban', 'text-muted-foreground hover:text-heading-foreground' => $view !== 'kanban'])
        >
            <x-tabler-layout-kanban class="w-4" /> {{ __('Board') }}
        </a>
        <a
            href="{{ route('dashboard.crm.leads.index', array_merge(request()->query(), ['view' => 'list'])) }}"
            @class(['flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm transition', 'bg-background shadow-xs text-heading-foreground' => $view === 'list', 'text-muted-foreground hover:text-heading-foreground' => $view !== 'list'])
        >
            <x-tabler-list class="w-4" /> {{ __('List') }}
        </a>
    </div>
    <x-button
        href="{{ route('dashboard.crm.leads.create') }}"
        size="sm"
        variant="primary"
    >
        <x-tabler-plus class="w-4" />
        {{ __('New Lead') }}
    </x-button>
@endsection

@section('content')
    <div class="py-10">

        @if ($view === 'kanban')
            {{-- ────────────── KANBAN BOARD ────────────── --}}

            @if ($statuses->isEmpty())
                <div class="py-20 text-center text-muted-foreground">
                    <x-tabler-layout-kanban class="mx-auto mb-4 w-12 opacity-30" />
                    <p>{{ __('No lead statuses configured yet.') }}</p>
                </div>
            @else
                {{-- Stats bar --}}
                <div class="mb-6 flex flex-wrap items-center gap-6 text-sm">
                    @php
                        $totalLeads = $statuses->flatMap->leads->count();
                        $totalValue = $statuses->flatMap->leads->sum('value');
                    @endphp
                    <div>
                        <span class="text-muted-foreground">{{ __('Total Leads') }}:</span>
                        <strong class="ms-1 text-heading-foreground">{{ $totalLeads }}</strong>
                    </div>
                    <div>
                        <span class="text-muted-foreground">{{ __('Pipeline Value') }}:</span>
                        <strong class="ms-1 text-heading-foreground">{{ number_format($totalValue, 0) }}</strong>
                    </div>
                </div>

                {{-- Board --}}
                <div
                    class="flex gap-4 overflow-x-auto pb-6"
                    id="leads-kanban"
                    data-move-url-base="{{ url('dashboard/crm/leads') }}"
                >
                    @foreach ($statuses as $status)
                        <div
                            class="flex w-72 shrink-0 flex-col rounded-2xl border border-border bg-card"
                            data-status-id="{{ $status->id }}"
                        >
                            {{-- Column header --}}
                            <div class="flex items-center justify-between border-b border-border px-4 py-3">
                                <div class="flex items-center gap-2">
                                    <span
                                        class="h-2.5 w-2.5 rounded-full"
                                        style="background-color: {{ $status->color }}"
                                    ></span>
                                    <span class="text-sm font-semibold text-heading-foreground">{{ $status->name }}</span>
                                    <span class="rounded-full bg-muted px-1.5 py-0.5 text-xs text-muted-foreground">{{ $status->leads->count() }}</span>
                                </div>
                                @if ($status->leads->sum('value') > 0)
                                    <span class="text-xs text-muted-foreground">{{ number_format($status->leads->sum('value'), 0) }}</span>
                                @endif
                            </div>

                            {{-- Cards --}}
                            <div
                                class="leads-column flex min-h-[200px] flex-col gap-2 p-3"
                                data-status-id="{{ $status->id }}"
                            >
                                @foreach ($status->leads as $lead)
                                    <div
                                        class="lead-card group cursor-pointer rounded-xl border border-border bg-background p-3 shadow-xs transition hover:border-primary/40 hover:shadow-sm"
                                        data-lead-id="{{ $lead->id }}"
                                        draggable="true"
                                    >
                                        <a
                                            href="{{ route('dashboard.crm.leads.show', $lead) }}"
                                            class="block"
                                        >
                                            <div class="mb-1.5 flex items-start justify-between gap-2">
                                                <span class="text-sm font-medium text-heading-foreground leading-snug">{{ $lead->title }}</span>
                                                @if ($lead->is_converted)
                                                    <span class="shrink-0 rounded-full bg-green-100 px-1.5 py-0.5 text-xs text-green-700">{{ __('Converted') }}</span>
                                                @endif
                                            </div>
                                            @if ($lead->contact_name)
                                                <p class="text-xs text-muted-foreground">{{ $lead->contact_name }}</p>
                                            @endif
                                            @if ($lead->company_name)
                                                <p class="text-xs text-muted-foreground">{{ $lead->company_name }}</p>
                                            @endif
                                            <div class="mt-2 flex items-center justify-between">
                                                @if ($lead->value > 0)
                                                    <span class="text-xs font-semibold text-heading-foreground">{{ $lead->currency }} {{ number_format($lead->value, 0) }}</span>
                                                @else
                                                    <span></span>
                                                @endif
                                                @if ($lead->source)
                                                    <span class="rounded-full bg-muted px-1.5 py-0.5 text-xs text-muted-foreground">{{ $lead->source->name }}</span>
                                                @endif
                                            </div>
                                        </a>
                                    </div>
                                @endforeach
                            </div>

                            {{-- Quick add --}}
                            <div class="border-t border-border px-3 py-2">
                                <a
                                    href="{{ route('dashboard.crm.leads.create', ['status_id' => $status->id]) }}"
                                    class="flex w-full items-center gap-1 rounded-lg px-2 py-1.5 text-xs text-muted-foreground hover:bg-muted hover:text-heading-foreground"
                                >
                                    <x-tabler-plus class="w-3.5" /> {{ __('Add lead') }}
                                </a>
                            </div>
                        </div>
                    @endforeach
                </div>
            @endif

        @else
            {{-- ────────────── LIST VIEW ────────────── --}}

            {{-- Filters --}}
            <form
                method="GET"
                class="mb-6 flex flex-wrap items-center gap-3"
            >
                <input
                    type="hidden"
                    name="view"
                    value="list"
                />
                <x-form.input
                    name="search"
                    placeholder="{{ __('Search title, name, email, company…') }}"
                    value="{{ request('search') }}"
                    class="w-64"
                />
                <select
                    name="status_id"
                    class="lqd-input lqd-input-md"
                >
                    <option value="">{{ __('All statuses') }}</option>
                    @foreach ($statuses as $s)
                        <option
                            value="{{ $s->id }}"
                            @selected(request('status_id') == $s->id)
                        >{{ $s->name }}</option>
                    @endforeach
                </select>
                <select
                    name="source_id"
                    class="lqd-input lqd-input-md"
                >
                    <option value="">{{ __('All sources') }}</option>
                    @foreach ($sources as $src)
                        <option
                            value="{{ $src->id }}"
                            @selected(request('source_id') == $src->id)
                        >{{ $src->name }}</option>
                    @endforeach
                </select>
                <x-button
                    type="submit"
                    size="sm"
                    variant="outline"
                >{{ __('Filter') }}</x-button>
                @if (request()->anyFilled(['search', 'status_id', 'source_id']))
                    <a
                        href="{{ route('dashboard.crm.leads.index', ['view' => 'list']) }}"
                        class="text-xs text-muted-foreground hover:underline"
                    >{{ __('Clear') }}</a>
                @endif
            </form>

            <div class="overflow-hidden rounded-2xl border border-border bg-card shadow-xs">
                <table class="w-full text-sm">
                    <thead class="border-b border-border bg-muted/40 text-left text-xs font-medium text-muted-foreground">
                        <tr>
                            <th class="px-4 py-3">{{ __('Title') }}</th>
                            <th class="px-4 py-3">{{ __('Contact') }}</th>
                            <th class="px-4 py-3">{{ __('Status') }}</th>
                            <th class="px-4 py-3">{{ __('Source') }}</th>
                            <th class="px-4 py-3">{{ __('Value') }}</th>
                            <th class="px-4 py-3">{{ __('Added') }}</th>
                            <th class="px-4 py-3 text-right">{{ __('Actions') }}</th>
                        </tr>
                    </thead>
                    <tbody class="divide-y divide-border">
                        @forelse ($leads as $lead)
                            <tr class="hover:bg-accent/30">
                                <td class="px-4 py-3">
                                    <a
                                        href="{{ route('dashboard.crm.leads.show', $lead) }}"
                                        class="font-medium text-heading-foreground hover:text-primary"
                                    >{{ $lead->title }}</a>
                                    @if ($lead->is_converted)
                                        <span class="ms-1 rounded-full bg-green-100 px-1.5 py-0.5 text-xs text-green-700">{{ __('Converted') }}</span>
                                    @endif
                                </td>
                                <td class="px-4 py-3">
                                    <div class="text-sm text-heading-foreground">{{ $lead->contact_name ?? '—' }}</div>
                                    @if ($lead->company_name)
                                        <div class="text-xs text-muted-foreground">{{ $lead->company_name }}</div>
                                    @endif
                                </td>
                                <td class="px-4 py-3">
                                    @if ($lead->status)
                                        <span
                                            class="rounded-full px-2 py-0.5 text-xs font-medium"
                                            style="background-color: {{ $lead->status->color }}22; color: {{ $lead->status->color }}"
                                        >{{ $lead->status->name }}</span>
                                    @else
                                        <span class="text-muted-foreground">—</span>
                                    @endif
                                </td>
                                <td class="px-4 py-3 text-xs text-muted-foreground">{{ $lead->source?->name ?? '—' }}</td>
                                <td class="px-4 py-3 text-sm font-medium text-heading-foreground">
                                    {{ $lead->value > 0 ? $lead->currency . ' ' . number_format($lead->value, 0) : '—' }}
                                </td>
                                <td class="px-4 py-3 text-xs text-muted-foreground">{{ $lead->created_at->format('M d, Y') }}</td>
                                <td class="px-4 py-3 text-right">
                                    <div class="flex items-center justify-end gap-2">
                                        <a
                                            href="{{ route('dashboard.crm.leads.edit', $lead) }}"
                                            class="text-muted-foreground hover:text-primary"
                                            title="{{ __('Edit') }}"
                                        >
                                            <x-tabler-pencil class="w-4" />
                                        </a>
                                        <form
                                            method="POST"
                                            action="{{ route('dashboard.crm.leads.destroy', $lead) }}"
                                            onsubmit="return confirm('{{ __('Delete this lead?') }}')"
                                        >
                                            @csrf
                                            @method('DELETE')
                                            <button
                                                type="submit"
                                                class="text-muted-foreground hover:text-red-500"
                                                title="{{ __('Delete') }}"
                                            >
                                                <x-tabler-trash class="w-4" />
                                            </button>
                                        </form>
                                    </div>
                                </td>
                            </tr>
                        @empty
                            <tr>
                                <td
                                    colspan="7"
                                    class="py-16 text-center text-muted-foreground"
                                >
                                    <x-tabler-target class="mx-auto mb-3 w-10 opacity-30" />
                                    <p>{{ __('No leads found.') }}</p>
                                    <a
                                        href="{{ route('dashboard.crm.leads.create') }}"
                                        class="mt-3 inline-flex items-center gap-1 text-sm text-primary hover:underline"
                                    >
                                        <x-tabler-plus class="w-4" /> {{ __('Add Lead') }}
                                    </a>
                                </td>
                            </tr>
                        @endforelse
                    </tbody>
                </table>
            </div>

            <div class="mt-4">
                {{ $leads->links() }}
            </div>
        @endif

    </div>
@endsection

@push('script')
    <script>
        // Simple drag-drop for kanban
        (function () {
            const board = document.getElementById('leads-kanban');
            if (!board) return;

            let dragging = null;

            board.querySelectorAll('.lead-card').forEach(card => {
                card.addEventListener('dragstart', e => {
                    dragging = card;
                    card.classList.add('opacity-50');
                });
                card.addEventListener('dragend', () => {
                    dragging = null;
                    card.classList.remove('opacity-50');
                });
            });

            board.querySelectorAll('.leads-column').forEach(col => {
                col.addEventListener('dragover', e => {
                    e.preventDefault();
                    const after = getDragAfterElement(col, e.clientY);
                    if (after == null) {
                        col.appendChild(dragging);
                    } else {
                        col.insertBefore(dragging, after);
                    }
                });

                col.addEventListener('drop', e => {
                    e.preventDefault();
                    const statusId = col.dataset.statusId;
                    const leadId   = dragging.dataset.leadId;
                    const order    = [...col.querySelectorAll('.lead-card')].indexOf(dragging);
                    const baseUrl  = board.dataset.moveUrlBase;

                    fetch(`${baseUrl}/${leadId}/move-status`, {
                        method: 'PATCH',
                        headers: {
                            'Content-Type': 'application/json',
                            'X-CSRF-TOKEN': document.querySelector('meta[name="csrf-token"]').content,
                        },
                        body: JSON.stringify({ status_id: statusId, order }),
                    });
                });
            });

            function getDragAfterElement(container, y) {
                const draggableElements = [...container.querySelectorAll('.lead-card:not(.opacity-50)')];
                return draggableElements.reduce((closest, child) => {
                    const box = child.getBoundingClientRect();
                    const offset = y - box.top - box.height / 2;
                    if (offset < 0 && offset > closest.offset) {
                        return { offset, element: child };
                    }
                    return closest;
                }, { offset: Number.NEGATIVE_INFINITY }).element;
            }
        })();
    </script>
@endpush
