@if ($errors->any())
    <div class="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
        <ul class="list-inside list-disc space-y-1">
            @foreach ($errors->all() as $error)<li>{{ $error }}</li>@endforeach
        </ul>
    </div>
@endif

<div class="rounded-2xl border border-border bg-card p-6 shadow-xs">
    <h3 class="mb-5 text-sm font-semibold text-heading-foreground">{{ __('Deal Details') }}</h3>
    <div class="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <x-form.group label="{{ __('Deal Title') }}" required class="sm:col-span-2">
            <x-form.input name="title" value="{{ old('title', $deal?->title) }}" required />
        </x-form.group>

        <x-form.group label="{{ __('Pipeline') }}" required>
            <select name="pipeline_id" id="pipeline_id" class="lqd-input lqd-input-md w-full" required>
                @foreach ($pipelines as $pipeline)
                    <option value="{{ $pipeline->id }}" @selected(old('pipeline_id', $deal?->pipeline_id ?? request('pipeline')) == $pipeline->id)
                        data-stages="{{ $pipeline->stages->map(fn($s) => ['id' => $s->id, 'name' => $s->name])->toJson() }}">
                        {{ $pipeline->name }}
                    </option>
                @endforeach
            </select>
        </x-form.group>

        <x-form.group label="{{ __('Stage') }}" required>
            <select name="stage_id" id="stage_id" class="lqd-input lqd-input-md w-full" required>
                @foreach ($pipelines->first()?->stages ?? [] as $stage)
                    <option value="{{ $stage->id }}" @selected(old('stage_id', $deal?->stage_id ?? request('stage')) == $stage->id)>{{ $stage->name }}</option>
                @endforeach
            </select>
        </x-form.group>

        <x-form.group label="{{ __('Value') }}">
            <x-form.input type="number" name="value" value="{{ old('value', $deal?->value ?? 0) }}" min="0" step="0.01" />
        </x-form.group>

        <x-form.group label="{{ __('Currency') }}" required>
            <x-form.input name="currency" value="{{ old('currency', $deal?->currency ?? 'USD') }}" maxlength="3" required />
        </x-form.group>

        <x-form.group label="{{ __('Probability (%)') }}">
            <x-form.input type="number" name="probability" value="{{ old('probability', $deal?->probability) }}" min="0" max="100" />
        </x-form.group>

        <x-form.group label="{{ __('Expected Close Date') }}">
            <x-form.input type="date" name="expected_close_date" value="{{ old('expected_close_date', $deal?->expected_close_date?->format('Y-m-d')) }}" />
        </x-form.group>

        @if ($deal)
            <x-form.group label="{{ __('Status') }}" required>
                <select name="status" class="lqd-input lqd-input-md w-full" required>
                    @foreach (['open', 'won', 'lost'] as $s)
                        <option value="{{ $s }}" @selected(old('status', $deal->status) === $s)>{{ ucfirst($s) }}</option>
                    @endforeach
                </select>
            </x-form.group>

            <x-form.group label="{{ __('Lost Reason') }}">
                <x-form.input name="lost_reason" value="{{ old('lost_reason', $deal?->lost_reason) }}" />
            </x-form.group>
        @endif
    </div>
</div>

<div class="rounded-2xl border border-border bg-card p-6 shadow-xs">
    <h3 class="mb-5 text-sm font-semibold text-heading-foreground">{{ __('Link to Contact / Company') }}</h3>
    <div class="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <x-form.group label="{{ __('Contact') }}">
            <select name="crm_contact_id" class="lqd-input lqd-input-md w-full">
                <option value="">{{ __('— None —') }}</option>
                @foreach ($contacts as $contact)
                    <option value="{{ $contact->id }}" @selected(old('crm_contact_id', $deal?->crm_contact_id ?? request('contact')) == $contact->id)>{{ $contact->full_name }}</option>
                @endforeach
            </select>
        </x-form.group>

        <x-form.group label="{{ __('Company') }}">
            <select name="crm_company_id" class="lqd-input lqd-input-md w-full">
                <option value="">{{ __('— None —') }}</option>
                @foreach ($companies as $company)
                    <option value="{{ $company->id }}" @selected(old('crm_company_id', $deal?->crm_company_id ?? request('crm_company_id')) == $company->id)>{{ $company->name }}</option>
                @endforeach
            </select>
        </x-form.group>
    </div>
</div>

<div class="rounded-2xl border border-border bg-card p-6 shadow-xs">
    <h3 class="mb-5 text-sm font-semibold text-heading-foreground">{{ __('Description') }}</h3>
    <textarea name="description" rows="3" class="lqd-input lqd-input-md w-full resize-none" placeholder="{{ __('Deal notes…') }}">{{ old('description', $deal?->description) }}</textarea>
</div>

@push('scripts')
<script>
// Dynamic stage options based on selected pipeline
document.addEventListener('DOMContentLoaded', function() {
    const pipelineSelect = document.getElementById('pipeline_id');
    const stageSelect    = document.getElementById('stage_id');

    if (!pipelineSelect || !stageSelect) return;

    pipelineSelect.addEventListener('change', function() {
        const option = this.options[this.selectedIndex];
        const stages = JSON.parse(option.dataset.stages || '[]');
        stageSelect.innerHTML = stages.map(s => `<option value="${s.id}">${s.name}</option>`).join('');
    });
});
</script>
@endpush
