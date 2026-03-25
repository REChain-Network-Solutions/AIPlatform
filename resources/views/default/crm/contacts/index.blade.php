@extends('panel.layout.app', ['disable_tblr' => true])

@section('title', __('Contacts — CRM'))

@section('titlebar_title', __('Contacts'))
@section('titlebar_subtitle', __('Manage your leads, prospects, and customers.'))

@section('titlebar_actions')
    <x-button
        href="{{ route('dashboard.crm.contacts.create') }}"
        size="sm"
        variant="primary"
    >
        <x-tabler-plus class="w-4" />
        {{ __('New Contact') }}
    </x-button>
@endsection

@section('content')
    <div class="py-10">

        {{-- Filters --}}
        <form
            method="GET"
            class="mb-6 flex flex-wrap items-center gap-3"
        >
            <x-form.input
                name="search"
                placeholder="{{ __('Search name, email, company…') }}"
                value="{{ request('search') }}"
                class="w-64"
            />
            <select
                name="status"
                class="lqd-input lqd-input-md"
            >
                <option value="">{{ __('All statuses') }}</option>
                @foreach (['lead', 'prospect', 'customer', 'churned', 'partner'] as $s)
                    <option
                        value="{{ $s }}"
                        @selected(request('status') === $s)
                    >{{ ucfirst($s) }}</option>
                @endforeach
            </select>
            <x-button
                type="submit"
                size="sm"
                variant="outline"
            >{{ __('Filter') }}</x-button>
            @if (request()->anyFilled(['search', 'status']))
                <a
                    href="{{ route('dashboard.crm.contacts.index') }}"
                    class="text-xs text-muted-foreground hover:underline"
                >{{ __('Clear') }}</a>
            @endif
        </form>

        {{-- Table --}}
        <div class="overflow-hidden rounded-2xl border border-border bg-card shadow-xs">
            <table class="w-full text-sm">
                <thead class="border-b border-border bg-muted/40 text-left text-xs font-medium text-muted-foreground">
                    <tr>
                        <th class="px-4 py-3">{{ __('Name') }}</th>
                        <th class="px-4 py-3">{{ __('Email') }}</th>
                        <th class="px-4 py-3">{{ __('Company') }}</th>
                        <th class="px-4 py-3">{{ __('Status') }}</th>
                        <th class="px-4 py-3">{{ __('Added') }}</th>
                        <th class="px-4 py-3 text-right">{{ __('Actions') }}</th>
                    </tr>
                </thead>
                <tbody class="divide-y divide-border">
                    @forelse ($contacts as $contact)
                        <tr class="hover:bg-accent/30">
                            <td class="px-4 py-3">
                                <a
                                    href="{{ route('dashboard.crm.contacts.show', $contact) }}"
                                    class="flex items-center gap-3 font-medium text-heading-foreground hover:text-primary"
                                >
                                    <div class="flex size-8 shrink-0 items-center justify-center rounded-full bg-primary/10 text-xs font-bold text-primary">
                                        {{ strtoupper(substr($contact->first_name, 0, 1)) }}{{ strtoupper(substr($contact->last_name ?? '', 0, 1)) }}
                                    </div>
                                    {{ $contact->full_name }}
                                </a>
                            </td>
                            <td class="px-4 py-3 text-muted-foreground">{{ $contact->email ?? '—' }}</td>
                            <td class="px-4 py-3 text-muted-foreground">{{ $contact->company_name ?? $contact->company?->name ?? '—' }}</td>
                            <td class="px-4 py-3">
                                @php
                                    $statusColors = ['lead' => 'bg-blue-100 text-blue-700', 'prospect' => 'bg-amber-100 text-amber-700', 'customer' => 'bg-green-100 text-green-700', 'churned' => 'bg-red-100 text-red-700', 'partner' => 'bg-violet-100 text-violet-700'];
                                @endphp
                                <span class="rounded-full px-2 py-0.5 text-xs font-medium {{ $statusColors[$contact->status] ?? 'bg-muted text-muted-foreground' }}">
                                    {{ ucfirst($contact->status) }}
                                </span>
                            </td>
                            <td class="px-4 py-3 text-xs text-muted-foreground">{{ $contact->created_at->format('M d, Y') }}</td>
                            <td class="px-4 py-3 text-right">
                                <div class="flex items-center justify-end gap-2">
                                    <a
                                        href="{{ route('dashboard.crm.contacts.edit', $contact) }}"
                                        class="text-muted-foreground hover:text-primary"
                                        title="{{ __('Edit') }}"
                                    >
                                        <x-tabler-pencil class="w-4" />
                                    </a>
                                    <form
                                        method="POST"
                                        action="{{ route('dashboard.crm.contacts.destroy', $contact) }}"
                                        onsubmit="return confirm('{{ __('Delete this contact?') }}')"
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
                                colspan="6"
                                class="py-16 text-center text-muted-foreground"
                            >
                                <x-tabler-users class="mx-auto mb-3 w-10 opacity-30" />
                                <p>{{ __('No contacts yet. Add your first contact to get started.') }}</p>
                                <a
                                    href="{{ route('dashboard.crm.contacts.create') }}"
                                    class="mt-3 inline-flex items-center gap-1 text-sm text-primary hover:underline"
                                >
                                    <x-tabler-plus class="w-4" /> {{ __('Add Contact') }}
                                </a>
                            </td>
                        </tr>
                    @endforelse
                </tbody>
            </table>
        </div>

        <div class="mt-4">
            {{ $contacts->links() }}
        </div>

    </div>
@endsection
