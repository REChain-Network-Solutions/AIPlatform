@php
	$base_class="inline-flex items-center leading-none text-base font-semibold px-1.5 py-0.5 leading-snug rounded-md"
@endphp

<span {{ $attributes->withoutTwMergeClasses()->twMerge($base_class, $attributes->get('class')) }}>
    <x-dynamic-component
        class="size-3"
        :component="$value <= 0 ? 'tabler-minus' : 'tabler-plus'"
    />
    {{ $value }}
</span>
