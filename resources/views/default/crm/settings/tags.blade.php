@extends('panel.layout.app', ['disable_tblr' => true])

@section('title', __('Tags — CRM Settings'))
@section('titlebar_title', __('Tags'))
@section('titlebar_subtitle', __('Categorise contacts, companies, and deals with tags.'))

@section('titlebar_actions')
    <x-button href="{{ route('dashboard.crm.settings.statuses') }}" size="sm" variant="outline">{{ __('Lead Statuses') }}</x-button>
    <x-button href="{{ route('dashboard.crm.settings.sources') }}" size="sm" variant="outline">{{ __('Lead Sources') }}</x-button>
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
                        <h3 class="text-sm font-semibold text-heading-foreground">{{ __('All Tags') }}</h3>
                    </div>
                    @if ($tags->isEmpty())
                        <div class="py-16 text-center text-muted-foreground">
                            <x-tabler-tag class="mx-auto mb-3 w-10 opacity-30" />
                            <p>{{ __('No tags yet. Add one to get started.') }}</p>
                        </div>
                    @else
                        <div class="flex flex-wrap gap-3 p-5">
                            @foreach ($tags as $tag)
                                <div class="group flex items-center gap-2 rounded-full border border-border bg-background px-3 py-1.5 text-sm">
                                    <span class="h-2.5 w-2.5 rounded-full shrink-0" style="background-color: {{ $tag->color }}"></span>
                                    <span class="font-medium text-heading-foreground">{{ $tag->name }}</span>
                                    <span class="text-xs text-muted-foreground">({{ $tag->contacts_count + $tag->companies_count + $tag->deals_count }})</span>
                                    <div class="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                                        <button type="button" class="text-muted-foreground hover:text-primary"
                                            onclick="openEditTag({{ $tag->id }}, '{{ addslashes($tag->name) }}', '{{ $tag->color }}')">
                                            <x-tabler-pencil class="w-3.5" />
                                        </button>
                                        <form method="POST" action="{{ route('dashboard.crm.settings.tags.destroy', $tag) }}" onsubmit="return confirm('{{ __('Delete tag?') }}')">
                                            @csrf @method('DELETE')
                                            <button type="submit" class="text-muted-foreground hover:text-red-500"><x-tabler-x class="w-3.5" /></button>
                                        </form>
                                    </div>
                                </div>
                            @endforeach
                        </div>
                    @endif
                </div>
            </div>

            <div class="lg:col-span-2">
                <div class="sticky top-6 rounded-2xl border border-border bg-card p-5 shadow-xs">
                    <h3 class="mb-4 text-sm font-semibold text-heading-foreground" id="tag-form-title">{{ __('Add Tag') }}</h3>
                    <form method="POST" id="tag-form" action="{{ route('dashboard.crm.settings.tags.store') }}">
                        @csrf
                        <input type="hidden" name="_method" id="tag-method" value="POST">
                        <div class="space-y-4">
                            <x-form.group label="{{ __('Tag Name') }}" required>
                                <x-form.input name="name" id="tag-name" placeholder="{{ __('e.g. VIP, Hot Lead, Partner') }}" required />
                            </x-form.group>
                            <x-form.group label="{{ __('Colour') }}">
                                <div class="flex items-center gap-3">
                                    <input type="color" name="color" id="tag-color" value="#6366f1" class="h-9 w-9 shrink-0 cursor-pointer rounded-lg border-0 p-0.5">
                                    <span class="text-xs text-muted-foreground">{{ __('Choose a label colour') }}</span>
                                </div>
                            </x-form.group>
                            <div class="flex items-center gap-2 pt-1">
                                <x-button type="submit" size="sm" variant="primary" id="tag-submit-btn">{{ __('Add Tag') }}</x-button>
                                <button type="button" id="tag-cancel-btn" class="hidden text-sm text-muted-foreground hover:text-heading-foreground" onclick="resetTagForm()">{{ __('Cancel') }}</button>
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
    function openEditTag(id, name, color) {
        document.getElementById('tag-form-title').textContent = '{{ __('Edit Tag') }}';
        document.getElementById('tag-form').action = `/dashboard/crm/settings/tags/${id}`;
        document.getElementById('tag-method').value = 'PUT';
        document.getElementById('tag-name').value = name;
        document.getElementById('tag-color').value = color;
        document.getElementById('tag-submit-btn').textContent = '{{ __('Save Changes') }}';
        document.getElementById('tag-cancel-btn').classList.remove('hidden');
    }
    function resetTagForm() {
        document.getElementById('tag-form-title').textContent = '{{ __('Add Tag') }}';
        document.getElementById('tag-form').action = '{{ route('dashboard.crm.settings.tags.store') }}';
        document.getElementById('tag-method').value = 'POST';
        document.getElementById('tag-name').value = '';
        document.getElementById('tag-color').value = '#6366f1';
        document.getElementById('tag-submit-btn').textContent = '{{ __('Add Tag') }}';
        document.getElementById('tag-cancel-btn').classList.add('hidden');
    }
</script>
@endpush
