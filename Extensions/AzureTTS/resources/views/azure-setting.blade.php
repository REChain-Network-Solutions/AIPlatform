@php
    $azure_regions_array_with_name_key = [
        'australiaeast' => 'Australia East (australiaeast)',
        'brazilsouth' => 'Brazil South (brazilsouth)',
        'canadacentral' => 'Canada Central (canadacentral)',
        'centralus' => 'Central US (centralus)',
        'eastasia' => 'East Asia (eastasia)',
        'eastus' => 'East US (eastus)',
        'eastus2' => 'East US 2 (eastus2)',
        'francecentral' => 'France Central (francecentral)',
        'centralindia' => 'India Central (centralindia)',
        'japaneast' => 'Japan East (japaneast)',
        'japanwest' => 'Japan West (japanwest)',
        'koreacentral' => 'Korea Central (koreacentral)',
        'northcentralus' => 'North Central US (northcentralus)',
        'northeurope' => 'North Europe (northeurope)',
        'southcentralus' => 'South Central US (southcentralus)',
        'southeastasia' => 'Southeast Asia (southeastasia)',
        'uksouth' => 'UK South (uksouth)',
        'westcentralus' => 'West Central US (westcentralus)',
        'westeurope' => 'West Europe (westeurope)',
        'westus' => 'West US (westus)',
        'westus2' => 'West US 2 (westus2)',
    ];
@endphp
<x-card
        class="mb-3 w-full"
        size="sm"
>
    <div class="mb-3">
        <label class="form-check form-switch">
            <input
                    class="form-check-input"
                    id="feature_tts_azure"
                    type="checkbox"
                    {{ $app_is_demo ? 'disabled' : '' }}
                    {{ setting('feature_tts_azure', false) ? 'checked' : '' }}
            >
            <span class="form-check-label">{{ __('Azure') }}</span>
        </label>
    </div>
    <div class="mb-3">
        <label class="form-label">{{ __('Azure API Key') }}</label>
        <input
                class="form-control"
                id="azure_api_key"
                type="text"
                name="azure_api_key"
                {{ $app_is_demo ? 'disabled' : '' }}
                placeholder="Azure API Key"
                value="{{ $app_is_demo ? '******************' : setting('azure_api_key', '') }}"
        >
    </div>
    {{-- azure region --}}
    <div class="mb-3">
        <label class="form-label">{{ __('Azure Region') }}</label>
        <select
                class="form-select"
                id="azure_region"
                name="azure_region"
                {{ $app_is_demo ? 'disabled' : '' }}
        >
            @foreach ($azure_regions_array_with_name_key as $key => $value)
                <option
                        value="{{ $key }}"
                        {{ setting('azure_region', 'eastus') === $key ? 'selected' : '' }}
                >
                    {{ $value }}
                </option>
            @endforeach
        </select>
    </div>

</x-card>