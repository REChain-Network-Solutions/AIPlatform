<div
    data-price="{{ $related['price'] }}"
    data-installed="{{ false }}"
    data-name="{{ $related['name'] }}"
    @class([
        'lqd-extension h-full rounded-[calc(var(--card-rounded)+3px)] transition-transform hover:-translate-y-1',
        'p-[3px] bg-gradient-to-r from-gradient-from via-gradient-via to-gradient-to' =>
            $related['is_featured'],
    ])
>
    <x-card
        class="relative flex h-full flex-col rounded-[17px] transition-all hover:shadow-lg"
        class:body="flex flex-col"
    >
        @if (trim($related['badge'], ' ') != '' || $related['price'] == 0)
            <p class="absolute end-5 top-5 m-0 rounded bg-[#FFF1DB] px-2 py-1 text-4xs font-semibold uppercase leading-tight tracking-widest text-[#242425]">
                @if (trim($related['badge'], ' ') != '')
                    {{ $related['badge'] }}
                @elseif ($related['price'] == 0)
                    @lang('Free')
                @endif
            </p>
        @endif

        <div @class([
            'mb-6 flex items-center',
            'size-[53px]' => $item['type'] === 'single',
        ])>
            <img
                @class([
                    'w-full h-auto rounded-xl' => $item['type'] === 'bundle',
                ])
                src="{{ $related['icon'] }}"
                @if ($item['type'] === 'single') width="53"
                height="53" @endif
                alt="{{ $related['name'] }}"
            >
        </div>

        <div class="mb-7 flex flex-wrap items-center gap-2">
            <h3 class="m-0 text-xl font-semibold">
                {{ $related['name'] }}
            </h3>
            <p class="review m-0 flex items-center gap-1 text-sm font-medium text-heading-foreground">
                <x-tabler-star-filled class="size-3" />
                {{ number_format($related['review'], 1) }}
            </p>
        </div>
        <p class="mb-7 text-base leading-normal">
            {{ $related['description'] }}
        </p>
        <a
            class="absolute inset-0 z-1"
            href="{{ route('dashboard.admin.marketplace.extension', ['slug' => $related['slug']]) }}"
        >
            <span class="sr-only">
                {{ __('View details') }}
            </span>
        </a>
        <div class="flex justify-between">
            @include('default.panel.admin.marketplace.particles.index-add-cart', ['item' => $related])
        </div>
    </x-card>
</div>
