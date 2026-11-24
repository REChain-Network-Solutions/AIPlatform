<div class="mt-9 flex w-full flex-col gap-4">
    @if (! $this->currentStepIs(4) || $executed)
        <x-button
            class="min-h-11 w-full"
            type="button"
            wire:click="nextStep"
            wire:offline.attr="disabled"
            wire:loading.attr="disabled"
            wire:target="nextStep"
            variant="secondary"
            :disabled="$this->currentStepIs(5)"
        >
            @if ($this->currentStepIs(5))
                @lang('Done')
            @else
                {{ $this->hasNextStep() ? __('Next') : __('Finish') }}
            @endif
            <span class="inline-grid size-7 place-content-center rounded-full bg-background text-foreground dark:bg-heading-foreground dark:text-header-background">
                <x-tabler-chevron-right class="size-4" />
            </span>
        </x-button>
    @endif

    <x-button
        class="min-h-11 w-full"
        type="button"
        wire:click="toPreviousStep"
        wire:offline.attr="disabled"
        wire:loading.attr="disabled"
        wire:target="toPreviousStep"
        :disabled="$this->currentStepIs(1) || $executed"
        variant="outline"
    >
        @lang('Back')
    </x-button>
</div>
