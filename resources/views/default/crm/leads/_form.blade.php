@if ($errors->any())
    <div class="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
        <ul class="list-inside list-disc space-y-1">
            @foreach ($errors->all() as $error)
                <li>{{ $error }}</li>
            @endforeach
        </ul>
    </div>
@endif

<div class="rounded-2xl border border-border bg-card p-6 shadow-xs">
    <h3 class="mb-5 text-sm font-semibold text-heading-foreground">{{ __('Lead Details') }}</h3>

    <div class="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <x-form.group
            label="{{ __('Lead Title') }}"
            class="sm:col-span-2"
            required
        >
            <x-form.input
                name="title"
                value="{{ old('title', $lead?->title) }}"
                placeholder="{{ __('e.g. Enterprise plan inquiry from Acme Corp') }}"
                required
            />
        </x-form.group>

        <x-form.group label="{{ __('Status') }}">
            <select
                name="crm_lead_status_id"
                class="lqd-input lqd-input-md w-full"
            >
                <option value="">{{ __('— None —') }}</option>
                @foreach ($statuses as $status)
                    <option
                        value="{{ $status->id }}"
                        @selected(old('crm_lead_status_id', $lead?->crm_lead_status_id ?? request('status_id')) == $status->id)
                    >{{ $status->name }}</option>
                @endforeach
            </select>
        </x-form.group>

        <x-form.group label="{{ __('Source') }}">
            <select
                name="crm_lead_source_id"
                class="lqd-input lqd-input-md w-full"
            >
                <option value="">{{ __('— Unknown —') }}</option>
                @foreach ($sources as $source)
                    <option
                        value="{{ $source->id }}"
                        @selected(old('crm_lead_source_id', $lead?->crm_lead_source_id) == $source->id)
                    >{{ $source->name }}</option>
                @endforeach
            </select>
        </x-form.group>

        <x-form.group label="{{ __('Value') }}">
            <x-form.input
                type="number"
                name="value"
                value="{{ old('value', $lead?->value ?? 0) }}"
                min="0"
                step="0.01"
            />
        </x-form.group>

        <x-form.group label="{{ __('Currency') }}">
            <x-form.input
                name="currency"
                value="{{ old('currency', $lead?->currency ?? 'USD') }}"
                maxlength="3"
                placeholder="USD"
            />
        </x-form.group>
    </div>
</div>

<div class="rounded-2xl border border-border bg-card p-6 shadow-xs">
    <h3 class="mb-5 text-sm font-semibold text-heading-foreground">{{ __('Contact Info') }}</h3>

    <div class="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <x-form.group label="{{ __('Contact Name') }}">
            <x-form.input
                name="contact_name"
                value="{{ old('contact_name', $lead?->contact_name) }}"
                placeholder="{{ __('Full name') }}"
            />
        </x-form.group>

        <x-form.group label="{{ __('Company') }}">
            <x-form.input
                name="company_name"
                value="{{ old('company_name', $lead?->company_name) }}"
                placeholder="{{ __('Company name') }}"
            />
        </x-form.group>

        <x-form.group label="{{ __('Email') }}">
            <x-form.input
                type="email"
                name="contact_email"
                value="{{ old('contact_email', $lead?->contact_email) }}"
            />
        </x-form.group>

        <x-form.group label="{{ __('Phone') }}">
            <x-form.input
                name="contact_phone"
                value="{{ old('contact_phone', $lead?->contact_phone) }}"
            />
        </x-form.group>

        <x-form.group
            label="{{ __('Website') }}"
            class="sm:col-span-2"
        >
            <x-form.input
                name="website"
                value="{{ old('website', $lead?->website) }}"
                placeholder="https://…"
            />
        </x-form.group>
    </div>
</div>

<div class="rounded-2xl border border-border bg-card p-6 shadow-xs">
    <h3 class="mb-5 text-sm font-semibold text-heading-foreground">{{ __('Notes') }}</h3>
    <textarea
        name="description"
        rows="4"
        class="lqd-input lqd-input-md w-full resize-none"
        placeholder="{{ __('Any details about this lead…') }}"
    >{{ old('description', $lead?->description) }}</textarea>
</div>
