<div
    x-data="updateAvailable({ routes: { check: '{{ route('dashboard.user.check.update-available') }}', appUpdate: '{{ route('dashboard.admin.update.index') }}', extensionUpdate: '{{ route('dashboard.admin.marketplace.index') }}' } })"
    x-cloak
    x-show="isAvailable"
>
    <x-dropdown.dropdown
        anchor="end"
        offsetY="10px"
    >
        <x-slot:trigger
            class="gap-2.5 text-[#56462E] shadow-none transition-none dark:text-foreground"
            ::href="route"
            @click.prevent="isVersionUpdateAvailable ? window.location.href = $el.href : null"
        >
            <span>{{ __('Update Available') }}</span>
            <x-tabler-circle-chevron-up stroke-width="1.5" />
        </x-slot:trigger>
        <x-slot:dropdown
            class="max-h-[40vh] overflow-hidden overflow-y-auto"
            x-bind:class="{ 'hidden': isVersionUpdateAvailable }"
        >
            <template x-for="item in updateAvailableExtensions">
                <a
                    class="flex items-center gap-2 border-b px-3 py-2 text-heading-foreground transition-colors last:border-b-0 hover:bg-foreground/5 hover:no-underline"
                    rel="alternate"
                    :href="'/dashboard/admin/marketplace/' + item.slug"
                >
                    <span x-text="item.name"></span>
                </a>
            </template>
        </x-slot:dropdown>
    </x-dropdown.dropdown>
</div>
