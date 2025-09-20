@props([
    'base_val' => 1,
    'dir' => null,
])

<div
    {{ $attributes->twMerge('lqd-progressive-blur-container overflow-hidden absolute top-0 left-0 right-0 bottom-0 pointer-events-none') }}
    style="--dir-y:{{ $dir === 'reverse' ? '1' : '-1' }}; --base-val:{{ $base_val }}"
>
    @for ($i = 0; $i < 7; $i++)
        <div class="lqd-progressive-blur-filter {{ $dir === 'reverse' ? 'reverse' : '' }} absolute bottom-0 left-0 right-0 top-0"></div>
    @endfor
    <div class="lqd-progressive-blur-gradient absolute bottom-0 left-0 right-0 top-0"></div>
</div>
