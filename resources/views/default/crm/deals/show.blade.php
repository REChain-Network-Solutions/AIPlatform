@extends('panel.layout.app', ['disable_tblr' => true])
@section('title', $deal->title . ' — CRM')
@section('titlebar_title', $deal->title)
@section('titlebar_back', route('dashboard.crm.deals.index'))
@section('titlebar_actions')
    <a href="{{ route('dashboard.crm.deals.edit', $deal) }}" class="lqd-btn lqd-btn-sm lqd-btn-outline">
        <x-tabler-pencil class="w-4" /> {{ __('Edit') }}
    </a>
@endsection

@section('content')
    <div class="py-10">
        <div class="grid grid-cols-1 gap-6 lg:grid-cols-3">
            {{-- Left --}}
            <div class="space-y-5">
                <div class="rounded-2xl border border-border bg-card p-6 shadow-xs">
                    @php $statusColors = ['open' => 'bg-blue-100 text-blue-700', 'won' => 'bg-green-100 text-green-700', 'lost' => 'bg-red-100 text-red-700']; @endphp
                    <span class="rounded-full px-2.5 py-0.5 text-xs font-medium {{ $statusColors[$deal->status] ?? '' }}">{{ ucfirst($deal->status) }}</span>

                    <div class="mt-4 text-3xl font-bold text-heading-foreground">{{ $deal->currency }} {{ number_format($deal->value, 0) }}</div>
                    <p class="mt-1 text-xs text-muted-foreground">{{ __('Deal Value') }}</p>

                    @if ($deal->stage)
                        <div class="mt-4">
                            <span class="rounded px-2 py-0.5 text-xs font-medium" style="background:{{ $deal->stage->color }}22;color:{{ $deal->stage->color }}">{{ $deal->stage->name }}</span>
                        </div>
                    @endif

                    <dl class="mt-5 space-y-2 text-sm">
                        @if ($deal->pipeline)
                            <div class="flex items-center gap-2 text-muted-foreground"><x-tabler-layout-kanban class="w-4" /> {{ $deal->pipeline->name }}</div>
                        @endif
                        @if ($deal->probability !== null)
                            <div class="flex items-center gap-2 text-muted-foreground"><x-tabler-percent class="w-4" /> {{ $deal->probability }}% {{ __('probability') }}</div>
                        @endif
                        @if ($deal->expected_close_date)
                            <div class="flex items-center gap-2 text-muted-foreground"><x-tabler-calendar class="w-4" /> {{ __('Close by') }}: {{ $deal->expected_close_date->format('M d, Y') }}</div>
                        @endif
                        @if ($deal->contact)
                            <div class="flex items-center gap-2 text-muted-foreground">
                                <x-tabler-user class="w-4" />
                                <a href="{{ route('dashboard.crm.contacts.show', $deal->contact) }}" class="hover:text-primary hover:underline">{{ $deal->contact->full_name }}</a>
                            </div>
                        @endif
                        @if ($deal->company)
                            <div class="flex items-center gap-2 text-muted-foreground">
                                <x-tabler-building class="w-4" />
                                <a href="{{ route('dashboard.crm.companies.show', $deal->company) }}" class="hover:text-primary hover:underline">{{ $deal->company->name }}</a>
                            </div>
                        @endif
                    </dl>
                </div>
            </div>

            {{-- Right: activities --}}
            <div class="col-span-2">
                <div class="rounded-2xl border border-border bg-card p-6 shadow-xs">
                    <h3 class="mb-4 text-sm font-semibold text-heading-foreground">{{ __('Log Activity') }}</h3>
                    <form method="POST" action="{{ route('dashboard.crm.activities.store') }}" class="mb-6 space-y-3">
                        @csrf
                        <input type="hidden" name="crm_deal_id" value="{{ $deal->id }}">
                        <div class="flex gap-3">
                            <select name="type" class="lqd-input lqd-input-md">
                                @foreach (['note', 'call', 'email', 'meeting', 'task', 'deadline'] as $t)
                                    <option value="{{ $t }}">{{ ucfirst($t) }}</option>
                                @endforeach
                            </select>
                            <x-form.input name="subject" placeholder="{{ __('Subject…') }}" class="flex-1" required />
                        </div>
                        <textarea name="body" rows="2" class="lqd-input lqd-input-md w-full resize-none" placeholder="{{ __('Details (optional)…') }}"></textarea>
                        <div class="flex items-center gap-3">
                            <x-form.input type="datetime-local" name="due_at" />
                            <x-button type="submit" size="sm" variant="outline">{{ __('Log') }}</x-button>
                        </div>
                    </form>

                    <h3 class="mb-3 text-sm font-semibold text-heading-foreground">{{ __('Activity Timeline') }}</h3>
                    <div class="space-y-3">
                        @forelse ($deal->activities as $activity)
                            <div class="flex items-start gap-3">
                                <div class="mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary">
                                    <x-dynamic-component :component="$activity->type_icon" class="w-3.5" />
                                </div>
                                <div class="flex-1">
                                    <p class="text-sm font-medium text-heading-foreground">{{ $activity->subject }}</p>
                                    @if ($activity->body)<p class="text-xs text-muted-foreground">{{ $activity->body }}</p>@endif
                                    <p class="mt-0.5 text-xs text-muted-foreground">{{ $activity->created_at->diffForHumans() }}</p>
                                </div>
                                @if ($activity->is_done)
                                    <x-tabler-circle-check class="w-4 shrink-0 text-green-500" />
                                @endif
                            </div>
                        @empty
                            <p class="text-sm text-muted-foreground">{{ __('No activities yet.') }}</p>
                        @endforelse
                    </div>
                </div>
            </div>
        </div>
    </div>
@endsection
