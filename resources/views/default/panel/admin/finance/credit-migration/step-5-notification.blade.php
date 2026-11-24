<div class="flex flex-col gap-6">
    <div class="flex flex-col gap-2">
        <h3 class="text-lg font-semibold text-heading-foreground">
            @lang('Notify Users')
        </h3>
        <p class="text-xs text-foreground/70">
            @lang('Optionally notify affected users about the migration to shared credits. You can use the newsletter feature to send a custom email.')
        </p>
    </div>

    <div class="rounded-card border border-card-border bg-card-background p-5">
        <div class="flex flex-col gap-4">
            <x-alert variant="info">
                @lang('Migration is complete! You can now notify your users about the change to shared credits using the Newsletter feature.')
            </x-alert>

            <div class="flex flex-col gap-2 text-xs">
                <div class="flex items-center justify-between border-b border-card-border pb-2">
                    <span class="text-foreground/70">@lang('Plans migrated')</span>
                    <span class="font-semibold text-heading-foreground">{{ $results['plans_migrated'] ?? 0 }}</span>
                </div>
                <div class="flex items-center justify-between border-b border-card-border pb-2">
                    <span class="text-foreground/70">@lang('Users migrated')</span>
                    <span class="font-semibold text-heading-foreground">{{ $results['users_migrated'] ?? 0 }}</span>
                </div>
            </div>

            <div class="flex flex-col gap-3 sm:flex-row">
                <x-button
                    href="{{ route('dashboard.email-templates.index') }}"
                    variant="primary"
                >
                    @lang('Go to Newsletter')
                    <x-tabler-arrow-right class="size-4" />
                </x-button>

                <x-button
                    href="{{ route('dashboard.admin.finance.plan.index') }}"
                    variant="outline"
                >
                    @lang('Back to Plans')
                </x-button>
            </div>
        </div>
    </div>
</div>
