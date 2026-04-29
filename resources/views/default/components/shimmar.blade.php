@php
    $classname = @twMerge('inline-flex items-center text-2xs font-medium text-heading-foreground/70', $attributes->get('class'));
@endphp

<span class="{{ $classname }}">
    <x-shimmer-text>{{ $slot->isNotEmpty() ? $slot : __('Thinking...') }}</x-shimmer-text>
</span>
