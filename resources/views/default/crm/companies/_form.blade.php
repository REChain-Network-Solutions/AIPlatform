@if ($errors->any())
    <div class="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
        <ul class="list-inside list-disc space-y-1">
            @foreach ($errors->all() as $error)<li>{{ $error }}</li>@endforeach
        </ul>
    </div>
@endif

<div class="rounded-2xl border border-border bg-card p-6 shadow-xs">
    <h3 class="mb-5 text-sm font-semibold text-heading-foreground">{{ __('Company Details') }}</h3>
    <div class="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <x-form.group label="{{ __('Company Name') }}" required class="sm:col-span-2">
            <x-form.input name="name" value="{{ old('name', $company?->name) }}" required />
        </x-form.group>

        <x-form.group label="{{ __('Domain') }}">
            <x-form.input name="domain" value="{{ old('domain', $company?->domain) }}" placeholder="acme.com" />
        </x-form.group>

        <x-form.group label="{{ __('Website') }}">
            <x-form.input name="website" value="{{ old('website', $company?->website) }}" placeholder="https://…" />
        </x-form.group>

        <x-form.group label="{{ __('Phone') }}">
            <x-form.input name="phone" value="{{ old('phone', $company?->phone) }}" />
        </x-form.group>

        <x-form.group label="{{ __('Industry') }}">
            <x-form.input name="industry" value="{{ old('industry', $company?->industry) }}" placeholder="{{ __('e.g. SaaS, Finance…') }}" />
        </x-form.group>

        <x-form.group label="{{ __('Type') }}">
            <select name="type" class="lqd-input lqd-input-md w-full">
                <option value="">{{ __('— None —') }}</option>
                @foreach (['prospect', 'customer', 'partner', 'vendor'] as $t)
                    <option value="{{ $t }}" @selected(old('type', $company?->type) === $t)>{{ ucfirst($t) }}</option>
                @endforeach
            </select>
        </x-form.group>

        <x-form.group label="{{ __('Employees') }}">
            <x-form.input type="number" name="employee_count" value="{{ old('employee_count', $company?->employee_count) }}" min="0" />
        </x-form.group>

        <x-form.group label="{{ __('Annual Revenue (USD)') }}">
            <x-form.input type="number" name="annual_revenue" value="{{ old('annual_revenue', $company?->annual_revenue) }}" min="0" step="0.01" />
        </x-form.group>
    </div>
</div>

<div class="rounded-2xl border border-border bg-card p-6 shadow-xs">
    <h3 class="mb-5 text-sm font-semibold text-heading-foreground">{{ __('Location') }}</h3>
    <div class="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <x-form.group label="{{ __('Address') }}" class="sm:col-span-2">
            <x-form.input name="address" value="{{ old('address', $company?->address) }}" />
        </x-form.group>
        <x-form.group label="{{ __('City') }}">
            <x-form.input name="city" value="{{ old('city', $company?->city) }}" />
        </x-form.group>
        <x-form.group label="{{ __('State') }}">
            <x-form.input name="state" value="{{ old('state', $company?->state) }}" />
        </x-form.group>
        <x-form.group label="{{ __('Country Code') }}">
            <x-form.input name="country" maxlength="2" placeholder="US" value="{{ old('country', $company?->country) }}" />
        </x-form.group>
    </div>
</div>

<div class="rounded-2xl border border-border bg-card p-6 shadow-xs">
    <h3 class="mb-5 text-sm font-semibold text-heading-foreground">{{ __('Description') }}</h3>
    <textarea name="description" rows="3" class="lqd-input lqd-input-md w-full resize-none" placeholder="{{ __('About this company…') }}">{{ old('description', $company?->description) }}</textarea>
</div>
