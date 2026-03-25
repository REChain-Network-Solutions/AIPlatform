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
    <h3 class="mb-5 text-sm font-semibold text-heading-foreground">{{ __('Basic Info') }}</h3>

    <div class="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <x-form.group label="{{ __('First Name') }}" required>
            <x-form.input
                name="first_name"
                value="{{ old('first_name', $contact?->first_name) }}"
                required
            />
        </x-form.group>

        <x-form.group label="{{ __('Last Name') }}">
            <x-form.input
                name="last_name"
                value="{{ old('last_name', $contact?->last_name) }}"
            />
        </x-form.group>

        <x-form.group label="{{ __('Email') }}">
            <x-form.input
                type="email"
                name="email"
                value="{{ old('email', $contact?->email) }}"
            />
        </x-form.group>

        <x-form.group label="{{ __('Phone') }}">
            <x-form.input
                name="phone"
                value="{{ old('phone', $contact?->phone) }}"
            />
        </x-form.group>

        <x-form.group label="{{ __('Job Title') }}">
            <x-form.input
                name="job_title"
                value="{{ old('job_title', $contact?->job_title) }}"
            />
        </x-form.group>

        <x-form.group label="{{ __('Status') }}" required>
            <select
                name="status"
                class="lqd-input lqd-input-md w-full"
                required
            >
                @foreach (['lead', 'prospect', 'customer', 'churned', 'partner'] as $s)
                    <option
                        value="{{ $s }}"
                        @selected(old('status', $contact?->status ?? 'lead') === $s)
                    >{{ ucfirst($s) }}</option>
                @endforeach
            </select>
        </x-form.group>
    </div>
</div>

<div class="rounded-2xl border border-border bg-card p-6 shadow-xs">
    <h3 class="mb-5 text-sm font-semibold text-heading-foreground">{{ __('Company') }}</h3>

    <div class="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <x-form.group label="{{ __('Company Name (freeform)') }}">
            <x-form.input
                name="company_name"
                value="{{ old('company_name', $contact?->company_name) }}"
                placeholder="{{ __('e.g. Acme Corp') }}"
            />
        </x-form.group>

        <x-form.group label="{{ __('Link to CRM Company') }}">
            <select
                name="crm_company_id"
                class="lqd-input lqd-input-md w-full"
            >
                <option value="">{{ __('— None —') }}</option>
                @foreach ($companies as $company)
                    <option
                        value="{{ $company->id }}"
                        @selected(old('crm_company_id', $contact?->crm_company_id) == $company->id)
                    >{{ $company->name }}</option>
                @endforeach
            </select>
        </x-form.group>
    </div>
</div>

<div class="rounded-2xl border border-border bg-card p-6 shadow-xs">
    <h3 class="mb-5 text-sm font-semibold text-heading-foreground">{{ __('Social & Web') }}</h3>

    <div class="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <x-form.group label="{{ __('LinkedIn URL') }}">
            <x-form.input
                name="linkedin_url"
                value="{{ old('linkedin_url', $contact?->linkedin_url) }}"
                placeholder="https://linkedin.com/in/…"
            />
        </x-form.group>

        <x-form.group label="{{ __('Website') }}">
            <x-form.input
                name="website"
                value="{{ old('website', $contact?->website) }}"
                placeholder="https://…"
            />
        </x-form.group>
    </div>
</div>

<div class="rounded-2xl border border-border bg-card p-6 shadow-xs">
    <h3 class="mb-5 text-sm font-semibold text-heading-foreground">{{ __('Location') }}</h3>

    <div class="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <x-form.group
            label="{{ __('Address') }}"
            class="sm:col-span-2"
        >
            <x-form.input
                name="address"
                value="{{ old('address', $contact?->address) }}"
            />
        </x-form.group>

        <x-form.group label="{{ __('City') }}">
            <x-form.input
                name="city"
                value="{{ old('city', $contact?->city) }}"
            />
        </x-form.group>

        <x-form.group label="{{ __('State / Region') }}">
            <x-form.input
                name="state"
                value="{{ old('state', $contact?->state) }}"
            />
        </x-form.group>

        <x-form.group label="{{ __('Country Code') }}">
            <x-form.input
                name="country"
                maxlength="2"
                placeholder="US"
                value="{{ old('country', $contact?->country) }}"
            />
        </x-form.group>
    </div>
</div>

<div class="rounded-2xl border border-border bg-card p-6 shadow-xs">
    <h3 class="mb-5 text-sm font-semibold text-heading-foreground">{{ __('Notes') }}</h3>
    <textarea
        name="notes"
        rows="4"
        class="lqd-input lqd-input-md w-full resize-none"
        placeholder="{{ __('Internal notes about this contact…') }}"
    >{{ old('notes', $contact?->notes) }}</textarea>
</div>
