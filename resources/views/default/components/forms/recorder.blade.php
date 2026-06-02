@php
    $base_class = 'lqd-recorder';
@endphp

<div
    {{ $attributes->twMerge($base_class) }}
    x-data="liquidRecorder({
        maxDurationSeconds: {{ $maxDurationSeconds !== null ? (int) $maxDurationSeconds : 'null' }},
        maxSizeMb: {{ $maxSizeMb !== null ? (int) $maxSizeMb : 'null' }}
    })"
>
    @if ($label)
        <label class="lqd-recorder-label lqd-input-label mb-3 flex cursor-pointer items-center gap-2 text-2xs font-medium leading-none text-label">
            {{ $label }}
            @if ($tooltip)
                <x-info-tooltip>{{ $tooltip }}</x-info-tooltip>
            @endif
        </label>
    @endif

    <input
        class="hidden"
        type="file"
        name="{{ $name }}"
        x-ref="fileInput"
        accept="audio/*"
    />

    <div
        class="lqd-recorder-pill flex items-center gap-2 overflow-hidden whitespace-nowrap rounded-full border border-foreground/5 bg-background p-1.5 shadow-[0_2px_4px_hsl(0_0%_0%/3%),0_8px_16px_hsl(0_0%_0%/3%),0_20px_55px_hsl(0_0%_0%/5%)] transition">
        <button
            class="lqd-recorder-button group/recorder-trigger relative inline-grid size-16 shrink-0 grid-cols-1 place-items-center rounded-full border-[3px] transition"
            type="button"
            :class="{ 'active': recording }"
            :title="recording ? '{{ __('Stop') }}' : '{{ __('Record') }}'"
            @click.prevent="toggle()"
        >
            <span
                class="lqd-recorder-indicator absolute left-1/2 top-1/2 size-[52px] -translate-x-1/2 -translate-y-1/2 overflow-hidden rounded-full transition before:absolute before:left-1/2 before:top-1/2 before:size-full before:-translate-x-1/2 before:-translate-y-1/2 before:scale-110 before:rounded-lg before:bg-[#fc0033] before:transition before:duration-200 before:ease-out group-hover/recorder-trigger:scale-105 group-active/recorder-trigger:scale-95 group-[&.active]/recorder-trigger:before:rotate-90 group-[&.active]/recorder-trigger:before:scale-[0.55]"
            ></span>
        </button>

        <span
            class="lqd-recorder-status mx-auto max-w-full truncate text-xs font-medium opacity-65"
            x-show="state === 'idle'"
        >
            {{ $placeholder ?? __('Click Record to Start') }}
        </span>

        <canvas
            class="lqd-recorder-live h-6 grow"
            x-ref="liveCanvas"
            x-show="state === 'recording'"
            x-cloak
        ></canvas>

        <div
            class="lqd-recorder-playback h-6 grow"
            x-ref="playbackContainer"
            x-show="state === 'recorded'"
            x-cloak
        ></div>

        <span
            class="lqd-recorder-time mx-2 text-xs font-medium tabular-nums opacity-65"
            x-show="state !== 'idle'"
            x-cloak
            x-text="state === 'recording' ? formatDuration(elapsed) : formatDuration(duration)"
        ></span>

        <button
            class="lqd-recorder-play inline-grid size-9 shrink-0 place-items-center rounded-full bg-foreground/5 transition hover:bg-foreground/10"
            type="button"
            x-show="state === 'recorded'"
            x-cloak
            :title="isPlaying ? '{{ __('Pause') }}' : '{{ __('Play') }}'"
            @click.prevent="togglePlayback()"
        >
            <x-tabler-player-play
                class="size-4 fill-current"
                x-show="!isPlaying"
            />
            <x-tabler-player-pause
                class="size-4 fill-current"
                x-show="isPlaying"
                x-cloak
            />
        </button>

        <button
            class="lqd-recorder-clear me-1 inline-grid size-9 shrink-0 place-items-center rounded-full bg-foreground/5 transition hover:bg-red-500 hover:text-white"
            type="button"
            x-show="state === 'recorded'"
            x-cloak
            title="{{ __('Discard') }}"
            @click.prevent="clear()"
        >
            <x-tabler-x class="size-4" />
        </button>
    </div>

    <p
        class="lqd-recorder-error mt-2 text-2xs text-red-500"
        x-show="error"
        x-cloak
        x-text="error"
    ></p>
</div>

@pushOnce('script')
    <script src="{{ custom_theme_url('/assets/libs/wavesurfer/wavesurfer.js') }}"></script>
@endPushOnce
