@php
    $base_class = 'lqd-card text-card-foreground bg-card-background w-full transition-all group/card';
    $head_base_class = 'lqd-card-head border-b border-card-border px-6 py-3.5 relative transition-border';
    $body_base_class = 'lqd-card-body relative only:grow';
    $foot_base_class = 'lqd-card-foot border-t border-card-border relative transition-border';

    $variations = [
        'variant' => [
            'none' => 'lqd-card-variant-none',
            'solid' => 'lqd-card-solid rounded-card',
            'outline' => 'lqd-card-outline border border-card-border rounded-card',
            'shadow' => 'lqd-card-shadow rounded-card',
            'outline-shadow' => 'lqd-card-outline-shadow border border-card-border rounded-card',
        ],
        'size' => [
            'none' => 'lqd-card-size-none',
            'xs' => 'lqd-card-xs px-5 py-3',
            'sm' => 'lqd-card-sm p-4',
            'md' => 'lqd-card-md py-5 px-7',
            'lg' => 'lqd-card-lg py-8 px-10',
        ],
    ];

    $variant = isset($variations['variant'][$variant]) ? $variations['variant'][$variant] : $variations['variant'][Theme::getSetting('defaultVariations.card.variant', 'outline')];
    $size = isset($variations['size'][$size]) ? $variations['size'][$size] : $variations['size'][Theme::getSetting('defaultVariations.card.size', 'md')];
@endphp

<div {{ $attributes->withoutTwMergeClasses()->twMerge($base_class, $variant, $attributes->get('class')) }}>
    @if (!empty($head))
        <div {{ $attributes->twMergeFor('head', $head_base_class, $head->attributes->get('class')) }}>
            {{ $head }}
        </div>
    @endif
    <div {{ $attributes->twMergeFor('body', $body_base_class, $size) }}>
        {{ $slot }}
    </div>
    @if (!empty($foot))
        <div {{ $attributes->twMergeFor('foot', $foot_base_class, $foot->attributes->get('class')) }}>
            {{ $foot }}
        </div>
    @endif
</div>
