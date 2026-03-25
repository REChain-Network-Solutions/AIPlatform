@extends('panel.layout.app', ['disable_tblr' => true])

@section('title', $lead->title . ' — CRM')
@section('titlebar_title', $lead->title)
@section('titlebar_subtitle', $lead->contact_name . ($lead->company_name ? ' · ' . $lead->company_name : ''))
@section('titlebar_back', route('dashboard.crm.leads.index'))

@section('titlebar_actions')
    @if (! $lead->is_converted)
        <button
            type="button"
            class="lqd-btn lqd-btn-sm lqd-btn-ghost text-green-600 hover:bg-green-50"
            onclick="document.getElementById('convert-modal').classList.remove('hidden')"
        >
            <x-tabler-transfer class="w-4" /> {{ __('Convert') }}
        </button>
    @endif
    <a
        href="{{ route('dashboard.crm.leads.edit', $lead) }}"
        class="lqd-btn lqd-btn-sm lqd-btn-outline"
    >
        <x-tabler-pencil class="w-4" /> {{ __('Edit') }}
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
            class="lqd-btn lqd-btn-sm lqd-btn-ghost text-red-500 hover:bg-red-50"
        >
            <x-tabler-trash class="w-4" />
        </button>
    </form>
@endsection

@section('content')
    <div class="py-10">

        @if (session('success'))
            <div class="mb-6 rounded-xl border border-green-200 bg-green-50 px-4 py-3 text-sm text-green-700">
                {{ session('success') }}
            </div>
        @endif

        <div class="grid grid-cols-1 gap-6 lg:grid-cols-3">

            {{-- Left: lead info --}}
            <div class="space-y-5">
                <div class="rounded-2xl border border-border bg-card p-6 shadow-xs">

                    {{-- Status badge --}}
                    <div class="mb-4 flex items-center gap-2">
                        @if ($lead->status)
                            <span
                                class="rounded-full px-2.5 py-0.5 text-xs font-medium"
                                style="background-color: {{ $lead->status->color }}22; color: {{ $lead->status->color }}"
                            >{{ $lead->status->name }}</span>
                        @endif
                        @if ($lead->is_converted)
                            <span class="rounded-full bg-green-100 px-2.5 py-0.5 text-xs font-medium text-green-700">{{ __('Converted') }}</span>
                        @endif
                    </div>

                    <dl class="space-y-3 text-sm">
                        @if ($lead->contact_name)
                            <div class="flex items-start gap-2 text-muted-foreground">
                                <x-tabler-user class="mt-0.5 w-4 shrink-0" />
                                <span>{{ $lead->contact_name }}</span>
                            </div>
                        @endif
                        @if ($lead->contact_email)
                            <div class="flex items-center gap-2 text-muted-foreground">
                                <x-tabler-mail class="w-4 shrink-0" />
                                <a
                                    href="mailto:{{ $lead->contact_email }}"
                                    class="hover:text-primary hover:underline"
                                >{{ $lead->contact_email }}</a>
                            </div>
                        @endif
                        @if ($lead->contact_phone)
                            <div class="flex items-center gap-2 text-muted-foreground">
                                <x-tabler-phone class="w-4 shrink-0" />
                                <span>{{ $lead->contact_phone }}</span>
                            </div>
                        @endif
                        @if ($lead->company_name)
                            <div class="flex items-center gap-2 text-muted-foreground">
                                <x-tabler-building class="w-4 shrink-0" />
                                <span>{{ $lead->company_name }}</span>
                            </div>
                        @endif
                        @if ($lead->website)
                            <div class="flex items-center gap-2 text-muted-foreground">
                                <x-tabler-world class="w-4 shrink-0" />
                                <a
                                    href="{{ $lead->website }}"
                                    target="_blank"
                                    class="hover:text-primary hover:underline"
                                >{{ $lead->website }}</a>
                            </div>
                        @endif
                    </dl>
                </div>

                {{-- Value & Source --}}
                <div class="rounded-2xl border border-border bg-card p-6 shadow-xs">
                    <h3 class="mb-4 text-sm font-semibold text-heading-foreground">{{ __('Qualification') }}</h3>
                    <dl class="space-y-3 text-sm">
                        <div class="flex items-center justify-between">
                            <span class="text-muted-foreground">{{ __('Value') }}</span>
                            <span class="font-semibold text-heading-foreground">
                                {{ $lead->value > 0 ? $lead->currency . ' ' . number_format($lead->value, 2) : '—' }}
                            </span>
                        </div>
                        <div class="flex items-center justify-between">
                            <span class="text-muted-foreground">{{ __('Source') }}</span>
                            <span class="text-heading-foreground">{{ $lead->source?->name ?? '—' }}</span>
                        </div>
                        <div class="flex items-center justify-between">
                            <span class="text-muted-foreground">{{ __('Created') }}</span>
                            <span class="text-heading-foreground">{{ $lead->created_at->format('M d, Y') }}</span>
                        </div>
                    </dl>
                </div>

                {{-- Conversion info --}}
                @if ($lead->is_converted)
                    <div class="rounded-2xl border border-green-200 bg-green-50 p-6 shadow-xs">
                        <h3 class="mb-3 text-sm font-semibold text-green-800">{{ __('Converted') }}</h3>
                        <dl class="space-y-2 text-sm">
                            <div class="text-green-700">
                                {{ __('Converted on') }} {{ $lead->converted_at->format('M d, Y') }}
                            </div>
                            @if ($lead->convertedContact)
                                <div>
                                    <span class="text-muted-foreground">{{ __('Contact:') }}</span>
                                    <a
                                        href="{{ route('dashboard.crm.contacts.show', $lead->convertedContact) }}"
                                        class="ms-1 text-primary hover:underline"
                                    >{{ $lead->convertedContact->full_name }}</a>
                                </div>
                            @endif
                            @if ($lead->convertedDeal)
                                <div>
                                    <span class="text-muted-foreground">{{ __('Deal:') }}</span>
                                    <a
                                        href="{{ route('dashboard.crm.deals.show', $lead->convertedDeal) }}"
                                        class="ms-1 text-primary hover:underline"
                                    >{{ $lead->convertedDeal->title }}</a>
                                </div>
                            @endif
                        </dl>
                    </div>
                @endif
            </div>

            {{-- Right: description --}}
            <div class="lg:col-span-2 space-y-5">
                @if ($lead->description)
                    <div class="rounded-2xl border border-border bg-card p-6 shadow-xs">
                        <h3 class="mb-3 text-sm font-semibold text-heading-foreground">{{ __('Notes') }}</h3>
                        <div class="prose prose-sm max-w-none text-muted-foreground">
                            {!! nl2br(e($lead->description)) !!}
                        </div>
                    </div>
                @endif
            </div>

        </div>
    </div>

    {{-- ── Convert Modal ── --}}
    @if (! $lead->is_converted)
        <div
            id="convert-modal"
            class="fixed inset-0 z-50 hidden flex items-center justify-center bg-black/50 p-4"
        >
            <div class="w-full max-w-md rounded-2xl border border-border bg-background p-6 shadow-xl">
                <div class="mb-4 flex items-center justify-between">
                    <h2 class="text-base font-semibold text-heading-foreground">{{ __('Convert Lead') }}</h2>
                    <button
                        type="button"
                        onclick="document.getElementById('convert-modal').classList.add('hidden')"
                        class="text-muted-foreground hover:text-heading-foreground"
                    >
                        <x-tabler-x class="w-5" />
                    </button>
                </div>

                <form
                    method="POST"
                    action="{{ route('dashboard.crm.leads.convert', $lead) }}"
                >
                    @csrf
                    <p class="mb-4 text-sm text-muted-foreground">
                        {{ __('This will create a Contact from this lead. Optionally create a Deal at the same time.') }}
                    </p>

                    <label class="mb-4 flex items-center gap-2 text-sm">
                        <input
                            type="checkbox"
                            name="create_deal"
                            value="1"
                            id="create_deal_toggle"
                            class="rounded"
                        />
                        {{ __('Also create a Deal') }}
                    </label>

                    <div
                        id="deal-fields"
                        class="hidden space-y-3"
                    >
                        @php
                            $pipelines = \App\Models\Crm\CrmPipeline::where('user_id', auth()->id())->with('stages')->get();
                        @endphp
                        <div>
                            <label class="mb-1 block text-xs font-medium text-muted-foreground">{{ __('Pipeline') }}</label>
                            <select
                                name="pipeline_id"
                                id="pipeline_select"
                                class="lqd-input lqd-input-md w-full"
                            >
                                <option value="">{{ __('— Select pipeline —') }}</option>
                                @foreach ($pipelines as $pipeline)
                                    <option value="{{ $pipeline->id }}">{{ $pipeline->name }}</option>
                                @endforeach
                            </select>
                        </div>
                        <div>
                            <label class="mb-1 block text-xs font-medium text-muted-foreground">{{ __('Stage') }}</label>
                            <select
                                name="stage_id"
                                id="stage_select"
                                class="lqd-input lqd-input-md w-full"
                            >
                                <option value="">{{ __('— Select stage —') }}</option>
                            </select>
                        </div>
                    </div>

                    <div class="mt-5 flex items-center gap-3">
                        <x-button
                            type="submit"
                            variant="primary"
                        >{{ __('Convert') }}</x-button>
                        <button
                            type="button"
                            onclick="document.getElementById('convert-modal').classList.add('hidden')"
                            class="text-sm text-muted-foreground hover:underline"
                        >{{ __('Cancel') }}</button>
                    </div>
                </form>
            </div>
        </div>
    @endif
@endsection

@push('script')
    <script>
        // Toggle deal fields in convert modal
        const toggle = document.getElementById('create_deal_toggle');
        const dealFields = document.getElementById('deal-fields');
        const pipelineSelect = document.getElementById('pipeline_select');
        const stageSelect = document.getElementById('stage_select');

        if (toggle) {
            toggle.addEventListener('change', () => {
                dealFields.classList.toggle('hidden', !toggle.checked);
            });
        }

        // Populate stages when pipeline changes
        const pipelineStages = @json($pipelines ?? []);

        if (pipelineSelect) {
            pipelineSelect.addEventListener('change', () => {
                const selected = pipelineStages.find(p => p.id == pipelineSelect.value);
                stageSelect.innerHTML = '<option value="">— Select stage —</option>';
                if (selected) {
                    selected.stages.forEach(stage => {
                        const opt = document.createElement('option');
                        opt.value = stage.id;
                        opt.textContent = stage.name;
                        stageSelect.appendChild(opt);
                    });
                }
            });
        }
    </script>
@endpush
