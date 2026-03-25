@extends('panel.layout.app', ['disable_tblr' => true])
@section('title', $company->name . ' — CRM')
@section('titlebar_title', $company->name)
@section('titlebar_subtitle', $company->industry)
@section('titlebar_back', route('dashboard.crm.companies.index'))
@section('titlebar_actions')
    <a href="{{ route('dashboard.crm.companies.edit', $company) }}" class="lqd-btn lqd-btn-sm lqd-btn-outline">
        <x-tabler-pencil class="w-4" /> {{ __('Edit') }}
    </a>
@endsection

@section('content')
    <div class="py-10">
        <div class="grid grid-cols-1 gap-6 lg:grid-cols-3">
            {{-- Info --}}
            <div class="space-y-5">
                <div class="rounded-2xl border border-border bg-card p-6 shadow-xs">
                    <div class="mb-4 flex items-center gap-4">
                        <div class="flex size-14 items-center justify-center rounded-full bg-violet-100 text-lg font-bold text-violet-700">
                            {{ strtoupper(substr($company->name, 0, 2)) }}
                        </div>
                        <div>
                            <h2 class="font-semibold text-heading-foreground">{{ $company->name }}</h2>
                            @if ($company->type)
                                <span class="rounded-full bg-violet-100 px-2 py-0.5 text-xs text-violet-700">{{ ucfirst($company->type) }}</span>
                            @endif
                        </div>
                    </div>
                    <dl class="space-y-2 text-sm">
                        @if ($company->domain)
                            <div class="flex items-center gap-2 text-muted-foreground"><x-tabler-globe class="w-4" /> <a href="https://{{ $company->domain }}" target="_blank" class="hover:text-primary hover:underline">{{ $company->domain }}</a></div>
                        @endif
                        @if ($company->phone)
                            <div class="flex items-center gap-2 text-muted-foreground"><x-tabler-phone class="w-4" /> {{ $company->phone }}</div>
                        @endif
                        @if ($company->industry)
                            <div class="flex items-center gap-2 text-muted-foreground"><x-tabler-building class="w-4" /> {{ $company->industry }}</div>
                        @endif
                        @if ($company->employee_count)
                            <div class="flex items-center gap-2 text-muted-foreground"><x-tabler-users class="w-4" /> {{ number_format($company->employee_count) }} {{ __('employees') }}</div>
                        @endif
                        @if ($company->annual_revenue)
                            <div class="flex items-center gap-2 text-muted-foreground"><x-tabler-currency-dollar class="w-4" /> {{ number_format($company->annual_revenue, 0) }} {{ __('annual revenue') }}</div>
                        @endif
                    </dl>
                </div>

                <div class="grid grid-cols-2 gap-3">
                    <div class="rounded-xl border border-border bg-card p-4 text-center shadow-xs">
                        <p class="text-xl font-bold text-heading-foreground">{{ $company->contacts->count() }}</p>
                        <p class="text-xs text-muted-foreground">{{ __('Contacts') }}</p>
                    </div>
                    <div class="rounded-xl border border-border bg-card p-4 text-center shadow-xs">
                        <p class="text-xl font-bold text-heading-foreground">{{ $company->deals->where('status', 'open')->count() }}</p>
                        <p class="text-xs text-muted-foreground">{{ __('Open Deals') }}</p>
                    </div>
                </div>
            </div>

            {{-- Contacts + Deals --}}
            <div class="col-span-2 space-y-6">
                <div class="rounded-2xl border border-border bg-card p-6 shadow-xs">
                    <div class="mb-4 flex items-center justify-between">
                        <h3 class="text-sm font-semibold text-heading-foreground">{{ __('Contacts') }}</h3>
                        <a href="{{ route('dashboard.crm.contacts.create', ['crm_company_id' => $company->id]) }}" class="text-xs text-primary hover:underline">+ {{ __('Add') }}</a>
                    </div>
                    @forelse ($company->contacts as $contact)
                        <a href="{{ route('dashboard.crm.contacts.show', $contact) }}" class="flex items-center gap-3 border-b border-border py-2 last:border-0 hover:opacity-80">
                            <div class="flex size-7 shrink-0 items-center justify-center rounded-full bg-primary/10 text-xs font-bold text-primary">
                                {{ strtoupper(substr($contact->first_name, 0, 1)) }}{{ strtoupper(substr($contact->last_name ?? '', 0, 1)) }}
                            </div>
                            <div>
                                <p class="text-sm font-medium text-heading-foreground">{{ $contact->full_name }}</p>
                                <p class="text-xs text-muted-foreground">{{ $contact->job_title ?? ucfirst($contact->status) }}</p>
                            </div>
                        </a>
                    @empty
                        <p class="text-sm text-muted-foreground">{{ __('No contacts linked.') }}</p>
                    @endforelse
                </div>

                <div class="rounded-2xl border border-border bg-card p-6 shadow-xs">
                    <div class="mb-4 flex items-center justify-between">
                        <h3 class="text-sm font-semibold text-heading-foreground">{{ __('Deals') }}</h3>
                        <a href="{{ route('dashboard.crm.deals.create', ['crm_company_id' => $company->id]) }}" class="text-xs text-primary hover:underline">+ {{ __('New deal') }}</a>
                    </div>
                    @forelse ($company->deals as $deal)
                        <a href="{{ route('dashboard.crm.deals.show', $deal) }}" class="flex items-center justify-between border-b border-border py-3 last:border-0 hover:opacity-80">
                            <div>
                                <p class="font-medium text-heading-foreground">{{ $deal->title }}</p>
                                @if ($deal->stage)
                                    <span class="rounded px-1.5 py-0.5 text-xs" style="background:{{ $deal->stage->color }}22;color:{{ $deal->stage->color }}">{{ $deal->stage->name }}</span>
                                @endif
                            </div>
                            <div class="text-right">
                                <p class="font-semibold text-heading-foreground">{{ $deal->currency }} {{ number_format($deal->value, 0) }}</p>
                                <p class="text-xs text-muted-foreground">{{ ucfirst($deal->status) }}</p>
                            </div>
                        </a>
                    @empty
                        <p class="text-sm text-muted-foreground">{{ __('No deals linked.') }}</p>
                    @endforelse
                </div>
            </div>
        </div>
    </div>
@endsection
