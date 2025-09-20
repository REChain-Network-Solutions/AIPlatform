@php
    $wrap_base_class = 'lqd-table-wrap w-full max-w-full overflow-x-auto';
    $base_class = 'lqd-table w-full text-start overflow-x-auto [-webkit-overflow-scrolling:touch] max-w-full';
    $head_base_class = 'lqd-table-head border-b text-start text-4xs leading-tight uppercase tracking-wider font-medium text-label transition-border';
    $body_base_class = '[&_tr:not(:last-child)]:border-b';
    $foot_base_class = 'lqd-table-foot';

    $variations = [
        'variant' => [
            'none' => 'lqd-table-variant-none shadow-none p-0',
            'solid' => 'lqd-table-solid rounded-card bg-card-background pt-1',
            'outline' => 'lqd-table-outline rounded-card border border-card-border pt-1',
            'shadow' => 'lqd-table-shadow rounded-card shadow-card bg-card-background pt-1',
            'plain' => 'lqd-table-plain',
            'outline-shadow' => 'lqd-table-outline-shadow rounded-card border border-card-border pt-1 shadow-card bg-card-background',
        ],
    ];

    $variant = isset($variations['variant'][$variant]) ? $variations['variant'][$variant] : $variations['variant'][Theme::getSetting('defaultVariations.table.variant', 'outline')];
@endphp

<div {{ $attributes->withoutTwMergeClasses()->twMergeFor('wrap', $variant, $wrap_base_class) }}>
    <table {{ $attributes->twMerge($base_class, $attributes->get('class')) }}>
        @if (!empty($head))
            <thead
                {{ $attributes->twMergeFor('head', $head_base_class, $head->attributes->get('class')) }}
                {{ $head->attributes }}
            >
                {{ $head }}
            </thead>
        @endif
        @if (!empty($body))
            <tbody
                {{ $attributes->twMergeFor('body', $body_base_class, $body->attributes->get('class')) }}
                {{ $body->attributes }}
            >
                {{ $body }}
            </tbody>
        @endif
        @if (!empty($foot))
            <tfoot
                {{ $attributes->twMergeFor('foot', $foot_base_class, $foot->attributes->get('class')) }}
                {{ $foot->attributes }}
            >
                {{ $foot }}
            </tfoot>
        @endif
    </table>
</div>
