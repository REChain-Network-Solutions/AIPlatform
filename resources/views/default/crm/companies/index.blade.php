@extends('panel.layout.app', ['disable_tblr' => true])

@section('title', __('Companies — CRM'))
@section('titlebar_title', __('Companies'))
@section('titlebar_subtitle', __('Business accounts in your CRM.'))

@section('titlebar_actions')
    <x-button
        href="{{ route('dashboard.crm.companies.create') }}"
        size="sm"
        variant="primary"
    >
        <x-tabler-plus class="w-4" />
        {{ __('New Company') }}
    </x-button>
@endsection

@section('content')
    <div class="py-10">
        <form
            method="GET"
            class="mb-6 flex flex-wrap items-center gap-3"
        >
            <x-form.input
                name="search"
                placeholder="{{ __('Search name, domain, industry…') }}"
                value="{{ request('search') }}"
                class="w-64"
            />
            <x-button
                type="submit"
                size="sm"
                variant="outline"
            >{{ __('Filter') }}</x-button>
            @if (request()->filled('search'))
                <a
                    href="{{ route('dashboard.crm.companies.index') }}"
                    class="text-xs text-muted-foreground hover:underline"
                >{{ __('Clear') }}</a>
            @endif
        </form>

        <div class="overflow-hidden rounded-2xl border border-border bg-card shadow-xs">
            <table class="w-full text-sm">
                <thead class="border-b border-border bg-muted/40 text-left text-xs font-medium text-muted-foreground">
                    <tr>
                        <th class="px-4 py-3">{{ __('Company') }}</th>
                        <th class="px-4 py-3">{{ __('Industry') }}</th>
                        <th class="px-4 py-3">{{ __('Contacts') }}</th>
                        <th class="px-4 py-3">{{ __('Deals') }}</th>
                        <th class="px-4 py-3">{{ __('Added') }}</th>
                        <th class="px-4 py-3 text-right">{{ __('Actions') }}</th>
                    </tr>
                </thead>
                <tbody class="divide-y divide-border">
                    @forelse ($companies as $company)
                        <tr class="hover:bg-accent/30">
                            <td class="px-4 py-3">
                                <a
                                    href="{{ route('dashboard.crm.companies.show', $company) }}"
                                    class="flex items-center gap-3 font-medium text-heading-foreground hover:text-primary"
                                >
                                    <div class="flex size-8 shrink-0 items-center justify-center rounded-full bg-violet-100 text-xs font-bold text-violet-700">
                                        {{ strtoupper(substr($company->name, 0, 2)) }}
                                    </div>
                                    <div>
                                        <p>{{ $company->name }}</p>
                                        @if ($company->domain)
                                            <p class="text-xs text-muted-foreground">{{ $company->domain }}</p>
                                        @endif
                                    </div>
                                </a>
                            </td>
                            <td class="px-4 py-3 text-muted-foreground">{{ $company->industry ?? '—' }}</td>
                            <td class="px-4 py-3">{{ $company->contacts_count }}</td>
                            <td class="px-4 py-3">{{ $company->deals_count }}</td>
                            <td class="px-4 py-3 text-xs text-muted-foreground">{{ $company->created_at->format('M d, Y') }}</td>
                            <td class="px-4 py-3 text-right">
                                <div class="flex items-center justify-end gap-2">
                                    <a
                                        href="{{ route('dashboard.crm.companies.edit', $company) }}"
                                        class="text-muted-foreground hover:text-primary"
                                    ><x-tabler-pencil class="w-4" /></a>
                                    <form
                                        method="POST"
                                        action="{{ route('dashboard.crm.companies.destroy', $company) }}"
                                        onsubmit="return confirm('{{ __('Delete this company?') }}')"
                                    >
                                        @csrf
                                        @method('DELETE')
                                        <button
                                            type="submit"
                                            class="text-muted-foreground hover:text-red-500"
                                        ><x-tabler-trash class="w-4" /></button>
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
                                <x-tabler-building class="mx-auto mb-3 w-10 opacity-30" />
                                <p>{{ __('No companies yet.') }}</p>
                                <a
                                    href="{{ route('dashboard.crm.companies.create') }}"
                                    class="mt-3 inline-flex items-center gap-1 text-sm text-primary hover:underline"
                                >
                                    <x-tabler-plus class="w-4" /> {{ __('Add Company') }}
                                </a>
                            </td>
                        </tr>
                    @endforelse
                </tbody>
            </table>
        </div>

        <div class="mt-4">{{ $companies->links() }}</div>
    </div>
@endsection
