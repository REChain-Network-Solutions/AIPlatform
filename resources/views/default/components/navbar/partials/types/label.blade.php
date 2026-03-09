<x-navbar.item>
    <x-navbar.label title="{{ __(data_get($item, 'label')) }}" :badge="data_get($item, 'badge') ?? ''">
        {{ __(data_get($item, 'label')) }}
    </x-navbar.label>
</x-navbar.item>
