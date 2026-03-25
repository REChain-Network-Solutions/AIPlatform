@extends('panel.layout.app', ['disable_tblr' => true])

@section('title', $contact->full_name . ' — CRM')
@section('titlebar_title', $contact->full_name)
@section('titlebar_subtitle', $contact->job_title . ($contact->company_name ? ' · ' . $contact->company_name : ''))
@section('titlebar_back', route('dashboard.crm.contacts.index'))

@section('titlebar_actions')
    <a
        href="{{ route('dashboard.crm.contacts.edit', $contact) }}"
        class="lqd-btn lqd-btn-sm lqd-btn-outline"
    >
        <x-tabler-pencil class="w-4" /> {{ __('Edit') }}
    </a>
@endsection

@section('content')
    <div class="py-10">
        <div class="grid grid-cols-1 gap-6 lg:grid-cols-3">

            {{-- Left: details --}}
            <div class="space-y-5">
                {{-- Info card --}}
                <div class="rounded-2xl border border-border bg-card p-6 shadow-xs">
                    <div class="mb-5 flex items-center gap-4">
                        <div class="flex size-14 items-center justify-center rounded-full bg-primary/10 text-xl font-bold text-primary">
                            {{ strtoupper(substr($contact->first_name, 0, 1)) }}{{ strtoupper(substr($contact->last_name ?? '', 0, 1)) }}
                        </div>
                        <div>
                            <h2 class="font-semibold text-heading-foreground">{{ $contact->full_name }}</h2>
                            <p class="text-xs text-muted-foreground">{{ $contact->job_title ?? '' }}</p>
                        </div>
                    </div>

                    @php
                        $statusColors = ['lead' => 'bg-blue-100 text-blue-700', 'prospect' => 'bg-amber-100 text-amber-700', 'customer' => 'bg-green-100 text-green-700', 'churned' => 'bg-red-100 text-red-700', 'partner' => 'bg-violet-100 text-violet-700'];
                    @endphp
                    <span class="rounded-full px-2.5 py-0.5 text-xs font-medium {{ $statusColors[$contact->status] ?? '' }}">
                        {{ ucfirst($contact->status) }}
                    </span>

                    <dl class="mt-5 space-y-3 text-sm">
                        @if ($contact->email)
                            <div class="flex items-center gap-2 text-muted-foreground">
                                <x-tabler-mail class="w-4 shrink-0" />
                                <a
                                    href="mailto:{{ $contact->email }}"
                                    class="hover:text-primary hover:underline"
                                >{{ $contact->email }}</a>
                            </div>
                        @endif
                        @if ($contact->phone)
                            <div class="flex items-center gap-2 text-muted-foreground">
                                <x-tabler-phone class="w-4 shrink-0" />
                                <span>{{ $contact->phone }}</span>
                            </div>
                        @endif
                        @if ($contact->company_name || $contact->company)
                            <div class="flex items-center gap-2 text-muted-foreground">
                                <x-tabler-building class="w-4 shrink-0" />
                                @if ($contact->company)
                                    <a
                                        href="{{ route('dashboard.crm.companies.show', $contact->company) }}"
                                        class="hover:text-primary hover:underline"
                                    >{{ $contact->company->name }}</a>
                                @else
                                    <span>{{ $contact->company_name }}</span>
                                @endif
                            </div>
                        @endif
                        @if ($contact->linkedin_url)
                            <div class="flex items-center gap-2 text-muted-foreground">
                                <x-tabler-brand-linkedin class="w-4 shrink-0" />
                                <a
                                    href="{{ $contact->linkedin_url }}"
                                    target="_blank"
                                    class="hover:text-primary hover:underline"
                                >{{ __('LinkedIn') }}</a>
                            </div>
                        @endif
                    </dl>
                </div>

                {{-- Stats --}}
                <div class="grid grid-cols-2 gap-3">
                    <div class="rounded-xl border border-border bg-card p-4 text-center shadow-xs">
                        <p class="text-xl font-bold text-heading-foreground">{{ $openDealsCount }}</p>
                        <p class="text-xs text-muted-foreground">{{ __('Open Deals') }}</p>
                    </div>
                    <div class="rounded-xl border border-border bg-card p-4 text-center shadow-xs">
                        <p class="text-xl font-bold text-heading-foreground">${{ number_format($totalDealValue, 0) }}</p>
                        <p class="text-xs text-muted-foreground">{{ __('Pipeline Value') }}</p>
                    </div>
                </div>

                @if ($contact->notes)
                    <div class="rounded-2xl border border-border bg-card p-5 shadow-xs">
                        <h3 class="mb-2 text-xs font-semibold text-muted-foreground">{{ __('Notes') }}</h3>
                        <p class="text-sm text-foreground">{{ $contact->notes }}</p>
                    </div>
                @endif
            </div>

            {{-- Right: deals + activities --}}
            <div class="col-span-2 space-y-6">

                {{-- Deals --}}
                <div class="rounded-2xl border border-border bg-card p-6 shadow-xs">
                    <div class="mb-4 flex items-center justify-between">
                        <h3 class="text-sm font-semibold text-heading-foreground">{{ __('Deals') }}</h3>
                        <a
                            href="{{ route('dashboard.crm.deals.create', ['contact' => $contact->id]) }}"
                            class="text-xs text-primary hover:underline"
                        >+ {{ __('New deal') }}</a>
                    </div>
                    @forelse ($contact->deals as $deal)
                        <a
                            href="{{ route('dashboard.crm.deals.show', $deal) }}"
                            class="flex items-center justify-between border-b border-border py-3 last:border-0 hover:opacity-80"
                        >
                            <div>
                                <p class="font-medium text-heading-foreground">{{ $deal->title }}</p>
                                @if ($deal->stage)
                                    <span
                                        class="rounded px-1.5 py-0.5 text-xs"
                                        style="background:{{ $deal->stage->color }}22;color:{{ $deal->stage->color }}"
                                    >{{ $deal->stage->name }}</span>
                                @endif
                            </div>
                            <div class="text-right">
                                <p class="font-semibold text-heading-foreground">{{ $deal->currency }} {{ number_format($deal->value, 0) }}</p>
                                <p class="text-xs text-muted-foreground">{{ ucfirst($deal->status) }}</p>
                            </div>
                        </a>
                    @empty
                        <p class="text-sm text-muted-foreground">{{ __('No deals linked to this contact.') }}</p>
                    @endforelse
                </div>

                {{-- Activity log + add --}}
                <div class="rounded-2xl border border-border bg-card p-6 shadow-xs">
                    <h3 class="mb-4 text-sm font-semibold text-heading-foreground">{{ __('Log Activity') }}</h3>
                    <form
                        method="POST"
                        action="{{ route('dashboard.crm.activities.store') }}"
                        class="mb-6 space-y-3"
                    >
                        @csrf
                        <input
                            type="hidden"
                            name="crm_contact_id"
                            value="{{ $contact->id }}"
                        >
                        <div class="flex gap-3">
                            <select
                                name="type"
                                class="lqd-input lqd-input-md"
                            >
                                @foreach (['note', 'call', 'email', 'meeting', 'task', 'deadline'] as $t)
                                    <option value="{{ $t }}">{{ ucfirst($t) }}</option>
                                @endforeach
                            </select>
                            <x-form.input
                                name="subject"
                                placeholder="{{ __('Subject…') }}"
                                class="flex-1"
                                required
                            />
                        </div>
                        <textarea
                            name="body"
                            rows="2"
                            class="lqd-input lqd-input-md w-full resize-none"
                            placeholder="{{ __('Details (optional)…') }}"
                        ></textarea>
                        <div class="flex items-center gap-3">
                            <x-form.input
                                type="datetime-local"
                                name="due_at"
                            />
                            <x-button
                                type="submit"
                                size="sm"
                                variant="outline"
                            >{{ __('Log') }}</x-button>
                        </div>
                    </form>

                    <h3 class="mb-3 text-sm font-semibold text-heading-foreground">{{ __('Activity Timeline') }}</h3>
                    <div class="space-y-3">
                        @forelse ($contact->activities as $activity)
                            <div class="flex items-start gap-3">
                                <div class="mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary">
                                    <x-dynamic-component
                                        :component="$activity->type_icon"
                                        class="w-3.5"
                                    />
                                </div>
                                <div class="flex-1">
                                    <p class="text-sm font-medium text-heading-foreground">{{ $activity->subject }}</p>
                                    @if ($activity->body)
                                        <p class="text-xs text-muted-foreground">{{ $activity->body }}</p>
                                    @endif
                                    <p class="mt-0.5 text-xs text-muted-foreground">
                                        {{ $activity->created_at->diffForHumans() }}
                                        @if ($activity->due_at)
                                            · {{ __('Due') }}: {{ $activity->due_at->format('M d, Y H:i') }}
                                        @endif
                                    </p>
                                </div>
                                @if (! $activity->is_done)
                                    <button
                                        class="shrink-0 rounded-full border border-border p-1 text-xs text-muted-foreground hover:bg-green-50 hover:text-green-600"
                                        hx-patch="{{ route('dashboard.crm.activities.done', $activity) }}"
                                        title="{{ __('Mark done') }}"
                                    >
                                        <x-tabler-check class="w-3" />
                                    </button>
                                @else
                                    <x-tabler-circle-check
                                        class="w-4 shrink-0 text-green-500"
                                        title="{{ __('Done') }}"
                                    />
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
