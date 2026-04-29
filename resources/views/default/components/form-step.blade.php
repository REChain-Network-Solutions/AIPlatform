@php
    $base_class = 'lqd-form-step flex items-center gap-3 rounded-xl bg-primary/5 px-4 py-3 text-sm font-semibold dark:bg-primary/10';
    $step_base_class = 'lqd-form-step-num inline-flex size-6 shrink-0 items-center justify-center rounded-full bg-primary text-sm text-primary-foreground';

    if ($autoIncrement) {
        $step_base_class .= ' lqd-form-step-num-auto';
    }
@endphp

<h3
    {{ $attributes->withoutTwMergeClasses()->twMerge($base_class) }}
    {{ $attributes }}
>
    @if ($step != '' || $autoIncrement)
        <span {{ $attributes->twMergeFor('step', $step_base_class) }}>
            @if (!$autoIncrement)
                {{ $step }}
            @endif
        </span>
    @endif
    {{ $label }}
    {{ $slot }}
</h3>
