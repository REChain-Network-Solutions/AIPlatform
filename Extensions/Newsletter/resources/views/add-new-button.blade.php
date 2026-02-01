<x-button
    href="{{ $app_is_demo ? '#' : route('dashboard.newsletter.create') }}"
    onclick="{{ $app_is_demo ? 'return toastr.info(\'This feature is disabled in Demo version.\')' : '' }}"
    variant="primary"
>
    <x-tabler-plus class="size-4" />
    {{ __('Add New Template') }}
</x-button>
