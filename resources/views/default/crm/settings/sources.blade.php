@extends('panel.layout.app', ['disable_tblr' => true])

@section('title', __('Lead Sources — CRM Settings'))
@section('titlebar_title', __('Lead Sources'))
@section('titlebar_subtitle', __('Track where your leads come from.'))

@section('titlebar_actions')
    <x-button href="{{ route('dashboard.crm.settings.statuses') }}" size="sm" variant="outline">{{ __('Lead Statuses') }}</x-button>
    <x-button href="{{ route('dashboard.crm.settings.tags') }}" size="sm" variant="outline">{{ __('Tags') }}</x-button>
@endsection

@section('content')
    <div class="py-10">

        @if (session('success'))
            <div class="mb-6 rounded-xl border border-green-200 bg-green-50 px-4 py-3 text-sm text-green-700">{{ session('success') }}</div>
        @endif

        <div class="grid grid-cols-1 gap-6 lg:grid-cols-5">

            <div class="lg:col-span-3">
                <div class="overflow-hidden rounded-2xl border border-border bg-card shadow-xs">
                    <div class="border-b border-border px-5 py-4">
                        <h3 class="text-sm font-semibold text-heading-foreground">{{ __('Lead Sources') }}</h3>
                    </div>
                    <table class="w-full text-sm">
                        <thead class="border-b border-border bg-muted/40 text-left text-xs text-muted-foreground">
                            <tr>
                                <th class="px-5 py-3">{{ __('Name') }}</th>
                                <th class="px-5 py-3">{{ __('Status') }}</th>
                                <th class="px-5 py-3">{{ __('Leads') }}</th>
                                <th class="px-5 py-3 text-right">{{ __('Actions') }}</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-border">
                            @forelse ($sources as $source)
                                <tr class="hover:bg-accent/20">
                                    <td class="px-5 py-3 font-medium text-heading-foreground">{{ $source->name }}</td>
                                    <td class="px-5 py-3">
                                        @if ($source->is_active)
                                            <span class="rounded-full bg-green-100 px-2 py-0.5 text-xs text-green-700">{{ __('Active') }}</span>
                                        @else
                                            <span class="rounded-full bg-muted px-2 py-0.5 text-xs text-muted-foreground">{{ __('Inactive') }}</span>
                                        @endif
                                    </td>
                                    <td class="px-5 py-3 text-muted-foreground">{{ $source->leads->count() }}</td>
                                    <td class="px-5 py-3 text-right">
                                        <div class="flex items-center justify-end gap-2">
                                            <button type="button" class="text-muted-foreground hover:text-primary"
                                                onclick="openEditSource({{ $source->id }}, '{{ addslashes($source->name) }}', {{ $source->is_active ? 'true' : 'false' }})">
                                                <x-tabler-pencil class="w-4" />
                                            </button>
                                            @if ($source->leads->isEmpty())
                                                <form method="POST" action="{{ route('dashboard.crm.settings.sources.destroy', $source) }}" onsubmit="return confirm('{{ __('Delete this source?') }}')">
                                                    @csrf @method('DELETE')
                                                    <button type="submit" class="text-muted-foreground hover:text-red-500"><x-tabler-trash class="w-4" /></button>
                                                </form>
                                            @endif
                                        </div>
                                    </td>
                                </tr>
                            @empty
                                <tr><td colspan="4" class="py-12 text-center text-muted-foreground">{{ __('No sources yet.') }}</td></tr>
                            @endforelse
                        </tbody>
                    </table>
                </div>
            </div>

            <div class="lg:col-span-2">
                <div class="sticky top-6 rounded-2xl border border-border bg-card p-5 shadow-xs" id="source-form-card">
                    <h3 class="mb-4 text-sm font-semibold text-heading-foreground" id="source-form-title">{{ __('Add Source') }}</h3>
                    <form method="POST" id="source-form" action="{{ route('dashboard.crm.settings.sources.store') }}">
                        @csrf
                        <input type="hidden" name="_method" id="source-method" value="POST">
                        <div class="space-y-4">
                            <x-form.group label="{{ __('Source Name') }}" required>
                                <x-form.input name="name" id="source-name" placeholder="{{ __('e.g. Google Ads, Referral, Cold Call') }}" required />
                            </x-form.group>
                            <label class="flex items-center gap-2 text-sm text-muted-foreground cursor-pointer">
                                <input type="checkbox" name="is_active" id="source-active" value="1" class="lqd-checkbox" checked>
                                {{ __('Active (shown in lead forms)') }}
                            </label>
                            <div class="flex items-center gap-2 pt-1">
                                <x-button type="submit" size="sm" variant="primary" id="source-submit-btn">{{ __('Add Source') }}</x-button>
                                <button type="button" id="source-cancel-btn" class="hidden text-sm text-muted-foreground hover:text-heading-foreground" onclick="resetSourceForm()">{{ __('Cancel') }}</button>
                            </div>
                        </div>
                    </form>
                </div>
            </div>

        </div>
    </div>
@endsection

@push('script')
<script>
    function openEditSource(id, name, isActive) {
        document.getElementById('source-form-title').textContent = '{{ __('Edit Source') }}';
        document.getElementById('source-form').action = `/dashboard/crm/settings/sources/${id}`;
        document.getElementById('source-method').value = 'PUT';
        document.getElementById('source-name').value = name;
        document.getElementById('source-active').checked = isActive;
        document.getElementById('source-submit-btn').textContent = '{{ __('Save Changes') }}';
        document.getElementById('source-cancel-btn').classList.remove('hidden');
    }
    function resetSourceForm() {
        document.getElementById('source-form-title').textContent = '{{ __('Add Source') }}';
        document.getElementById('source-form').action = '{{ route('dashboard.crm.settings.sources.store') }}';
        document.getElementById('source-method').value = 'POST';
        document.getElementById('source-name').value = '';
        document.getElementById('source-active').checked = true;
        document.getElementById('source-submit-btn').textContent = '{{ __('Add Source') }}';
        document.getElementById('source-cancel-btn').classList.add('hidden');
    }
</script>
@endpush
