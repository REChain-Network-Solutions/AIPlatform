@php
    $base_class = 'lqd-dropdown flex relative group/dropdown [--dropdown-offset:0px]';
    $trigger_base_class = 'lqd-dropdown-trigger hover:translate-y-0
	before:absolute before:-inset-3 before:pointer-events-none
	focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2
	group-[&.lqd-is-active]/dropdown:before:pointer-events-auto';
    $dropdown_base_class = 'lqd-dropdown-dropdown absolute opacity-0 invisible z-[1000] translate-y-1 pointer-events-none transition
		before:absolute before:bottom-full before:-top-[--dropdown-offset] before:inset-x-0
		[&.lqd-is-active]/dropdown:opacity-100 [&.lqd-is-active]/dropdown:visible [&.lqd-is-active]/dropdown:translate-y-0 [&.lqd-is-active]/dropdown:pointer-events-auto';
    $dropdown_content_base_class =
        'lqd-dropdown-dropdown-content w-44 border border-dropdown-border rounded-dropdown bg-dropdown-background text-dropdown-foreground shadow-lg shadow-black/5';

    if ($teleport) {
        $dropdown_base_class .= ' top-0';
    } else {
        $dropdown_base_class .= ' top-full';
    }

    if ($anchor === 'start') {
        $dropdown_base_class .= ' start-0';
    } else {
        $dropdown_base_class .= ' end-0';
    }
@endphp

<div
    class="{{ @twMerge($base_class, $attributes->get('class')) }}"
    @style([
        '--dropdown-offset: ' . $offsetY . '' => !empty($offsetY),
    ])
    x-data="dropdown({ triggerType: '{{ $triggerType }}', preferredAnchor: '{{ $anchor }}', offsetY: '{{ $offsetY }}', teleport: {{ $teleport ? 'true' : 'false' }} })"
    x-bind="parent"
    x-ref="parent"
>
    <x-button
        class="{{ @twMerge($trigger_base_class, $attributes->get('class:trigger'), $trigger->attributes->get('class')) }}"
        variant="{{ $trigger->attributes->get('variant') ? $trigger->attributes->get('variant') : 'link' }}"
        x-bind="trigger"
        x-ref="trigger"
        :attributes="$trigger->attributes"
    >
        {{ $trigger }}
    </x-button>

    @if ($teleport)
        <template x-teleport="body">
    @endif
    <div
        class="{{ @twMerge($dropdown_base_class, $attributes->get('class:dropdown-dropdown')) }}"
        @style([
            '--dropdown-offset: ' . $offsetY . '' => !empty($offsetY),
        ])
        :class="{ 'lqd-is-active': open }"
        x-bind="dropdown"
        x-ref="dropdown"
    >
        <div
            class="{{ @twMerge($dropdown_content_base_class, $attributes->get('class:dropdown'), $dropdown->attributes->get('class')) }}"
            {{ $dropdown->attributes->only('x-bind:class') }}
        >
            {{ $dropdown }}
        </div>
    </div>
    @if ($teleport)
        </template>
    @endif
</div>
