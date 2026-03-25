@extends('panel.layout.app', ['disable_tblr' => true])

@section('title', __('Lead Statuses — CRM Settings'))
@section('titlebar_title', __('Lead Statuses'))
@section('titlebar_subtitle', __('Customise the stages in your lead pipeline.'))

@section('titlebar_actions')
    <x-button href="{{ route('dashboard.crm.settings.sources') }}" size="sm" variant="outline">
        {{ __('Lead Sources') }}
    </x-button>
    <x-button href="{{ route('dashboard.crm.settings.tags') }}" size="sm" variant="outline">
        {{ __('Tags') }}
    </x-button>
@endsection

@section('content')
    <div class="py-10">

        @if (session('success'))
            <div class="mb-6 rounded-xl border border-green-200 bg-green-50 px-4 py-3 text-sm text-green-700">{{ session('success') }}</div>
        @endif

        <div class="grid grid-cols-1 gap-6 lg:grid-cols-5">

            {{-- Status List --}}
            <div class="lg:col-span-3">
                <div class="overflow-hidden rounded-2xl border border-border bg-card shadow-xs">
                    <div class="border-b border-border px-5 py-4">
                        <h3 class="text-sm font-semibold text-heading-foreground">{{ __('Current Statuses') }}</h3>
                        <p class="mt-0.5 text-xs text-muted-foreground">{{ __('Drag to reorder. Leads will follow this order in the kanban view.') }}</p>
                    </div>

                    <ul id="statuses-sortable" class="divide-y divide-border">
                        @forelse ($statuses as $status)
                            <li class="flex items-center gap-3 px-5 py-3" data-id="{{ $status->id }}">
                                <x-tabler-grip-vertical class="w-4 shrink-0 cursor-grab text-muted-foreground/40" />
                                <span class="h-3 w-3 shrink-0 rounded-full" style="background-color: {{ $status->color }}"></span>
                                <span class="flex-1 text-sm font-medium text-heading-foreground">{{ $status->name }}</span>
                                <div class="flex items-center gap-1.5">
                                    @if ($status->is_default)
                                        <span class="rounded-full bg-primary/10 px-2 py-0.5 text-xs text-primary">{{ __('Default') }}</span>
                                    @endif
                                    @if ($status->is_final)
                                        <span class="rounded-full bg-muted px-2 py-0.5 text-xs text-muted-foreground">{{ __('Final') }}</span>
                                    @endif
                                </div>
                                <button
                                    type="button"
                                    class="text-muted-foreground hover:text-primary"
                                    onclick="openEditStatus({{ $status->id }}, '{{ addslashes($status->name) }}', '{{ $status->color }}', {{ $status->is_default ? 'true' : 'false' }}, {{ $status->is_final ? 'true' : 'false' }})"
                                >
                                    <x-tabler-pencil class="w-4" />
                                </button>
                                @if (!$status->leads()->exists())
                                    <form method="POST" action="{{ route('dashboard.crm.settings.statuses.destroy', $status) }}" onsubmit="return confirm('{{ __('Delete this status?') }}')">
                                        @csrf @method('DELETE')
                                        <button type="submit" class="text-muted-foreground hover:text-red-500"><x-tabler-trash class="w-4" /></button>
                                    </form>
                                @endif
                            </li>
                        @empty
                            <li class="px-5 py-10 text-center text-sm text-muted-foreground">{{ __('No statuses yet.') }}</li>
                        @endforelse
                    </ul>
                </div>
            </div>

            {{-- Add / Edit Form --}}
            <div class="lg:col-span-2">
                <div class="sticky top-6 rounded-2xl border border-border bg-card p-5 shadow-xs" id="status-form-card">
                    <h3 class="mb-4 text-sm font-semibold text-heading-foreground" id="status-form-title">{{ __('Add Status') }}</h3>
                    <form method="POST" id="status-form" action="{{ route('dashboard.crm.settings.statuses.store') }}">
                        @csrf
                        <input type="hidden" name="_method" id="status-method" value="POST">
                        <input type="hidden" name="_status_id" id="status-id" value="">

                        <div class="space-y-4">
                            <x-form.group label="{{ __('Name') }}" required>
                                <x-form.input name="name" id="status-name" placeholder="{{ __('e.g. Qualified') }}" required />
                            </x-form.group>

                            <x-form.group label="{{ __('Colour') }}">
                                <div class="flex items-center gap-3">
                                    <input type="color" name="color" id="status-color" value="#6366f1" class="h-9 w-9 shrink-0 cursor-pointer rounded-lg border-0 p-0.5">
                                    <span class="text-xs text-muted-foreground">{{ __('Pick a colour for the kanban column header') }}</span>
                                </div>
                            </x-form.group>

                            <div class="flex flex-col gap-2">
                                <label class="flex items-center gap-2 text-sm text-muted-foreground cursor-pointer">
                                    <input type="checkbox" name="is_default" id="status-default" value="1" class="lqd-checkbox">
                                    {{ __('Set as default status') }}
                                </label>
                                <label class="flex items-center gap-2 text-sm text-muted-foreground cursor-pointer">
                                    <input type="checkbox" name="is_final" id="status-final" value="1" class="lqd-checkbox">
                                    {{ __('This is a final/closed status') }}
                                </label>
                            </div>

                            <div class="flex items-center gap-2 pt-1">
                                <x-button type="submit" size="sm" variant="primary" id="status-submit-btn">{{ __('Add Status') }}</x-button>
                                <button type="button" id="status-cancel-btn" class="hidden text-sm text-muted-foreground hover:text-heading-foreground" onclick="resetStatusForm()">{{ __('Cancel') }}</button>
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
    function openEditStatus(id, name, color, isDefault, isFinal) {
        document.getElementById('status-form-title').textContent = '{{ __('Edit Status') }}';
        document.getElementById('status-form').action = `/dashboard/crm/settings/statuses/${id}`;
        document.getElementById('status-method').value = 'PUT';
        document.getElementById('status-id').value = id;
        document.getElementById('status-name').value = name;
        document.getElementById('status-color').value = color;
        document.getElementById('status-default').checked = isDefault;
        document.getElementById('status-final').checked = isFinal;
        document.getElementById('status-submit-btn').textContent = '{{ __('Save Changes') }}';
        document.getElementById('status-cancel-btn').classList.remove('hidden');
        document.getElementById('status-form-card').scrollIntoView({ behavior: 'smooth' });
    }

    function resetStatusForm() {
        document.getElementById('status-form-title').textContent = '{{ __('Add Status') }}';
        document.getElementById('status-form').action = '{{ route('dashboard.crm.settings.statuses.store') }}';
        document.getElementById('status-method').value = 'POST';
        document.getElementById('status-id').value = '';
        document.getElementById('status-name').value = '';
        document.getElementById('status-color').value = '#6366f1';
        document.getElementById('status-default').checked = false;
        document.getElementById('status-final').checked = false;
        document.getElementById('status-submit-btn').textContent = '{{ __('Add Status') }}';
        document.getElementById('status-cancel-btn').classList.add('hidden');
    }

    // Drag-to-reorder
    (function () {
        const list = document.getElementById('statuses-sortable');
        if (!list) return;
        let dragging = null;

        list.querySelectorAll('li').forEach(li => {
            li.draggable = true;
            li.addEventListener('dragstart', () => { dragging = li; li.classList.add('opacity-50'); });
            li.addEventListener('dragend', () => {
                li.classList.remove('opacity-50');
                const ids = [...list.querySelectorAll('li')].map(el => el.dataset.id);
                fetch('{{ route('dashboard.crm.settings.statuses.reorder') }}', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', 'X-CSRF-TOKEN': document.querySelector('meta[name="csrf-token"]').content },
                    body: JSON.stringify({ ids }),
                });
            });
            li.addEventListener('dragover', e => {
                e.preventDefault();
                const rect = li.getBoundingClientRect();
                if (e.clientY < rect.top + rect.height / 2) list.insertBefore(dragging, li);
                else list.insertBefore(dragging, li.nextSibling);
            });
        });
    })();
</script>
@endpush
