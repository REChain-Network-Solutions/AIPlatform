@aware(['selectedValue' => null])

@php
    $base_class = 'lqd-selectbox-option flex w-full items-center justify-normal gap-2.5 overflow-hidden rounded-lg px-4 py-2.5 text-start text-2xs font-medium text-heading-foreground transition hover:bg-foreground/5';
    $isSelected = $selectedValue !== null && (string) $selectedValue === (string) $value;
@endphp

<button
    type="button"
    {{ $attributes->twMerge($base_class, $isSelected ? 'bg-foreground/5' : '') }}
    x-init="registerOption({{ json_encode($value) }}, {{ json_encode($label) }})"
    x-show="matches({{ json_encode($value) }}, {{ json_encode($label) }})"
    :class="{ 'bg-foreground/5': selected === {{ json_encode($value) }} }"
    @click.prevent="select({{ json_encode($value) }}); toggle('collapse')"
>
    {{ $slot }}

    <template x-if="selected === {{ json_encode($value) }}">
        <x-tabler-check class="ms-auto size-5 shrink-0" />
    </template>
</button>
