<div
    data-price="{{ $item['price'] }}"
    data-installed="{{ $item['installed'] }}"
    data-name="{{ $item['name'] }}"
    @class([
        'lqd-extension h-full rounded-[calc(var(--card-rounded)+3px)] transition-transform hover:-translate-y-1',
        'p-[3px] bg-gradient-to-r from-gradient-from via-gradient-via to-gradient-to' =>
            $item['is_featured'],
    ])
>
    <x-card
        class="relative flex h-full flex-col rounded-[17px] transition-all hover:shadow-lg"
        class:body="flex flex-col"
    >
        @if (trim($item['badge'], ' ') != '' || $item['price'] == 0)
            <p class="absolute end-5 top-5 m-0 rounded bg-[#FFF1DB] px-2 py-1 text-4xs font-semibold uppercase leading-tight tracking-widest text-[#242425]">
                @if (trim($item['badge'], ' ') != '')
                    {{ $item['badge'] }}
                @elseif ($item['price'] == 0)
                    @lang('Free')
                @endif
            </p>
        @endif

        @if (isset($item['db_version']) && $item['version'] != $item['db_version'] && $item['installed'])
            <p
                class="top-{{ $item['price'] == 0 ? '10' : '5' }} absolute end-5 m-0 rounded bg-purple-50 px-2 py-1 text-4xs font-semibold uppercase leading-tight tracking-widest text-[#242425] text-purple-700 ring-1 ring-inset ring-purple-700/10">
                <a href="{{ route('dashboard.admin.marketplace.liextension') }}">{{ __('Update Available') }}</a>
            </p>
        @endif
        @if ($item['youtube'])
            <div
                class="absolute end-5 top-16 z-[1000000000000000] cursor-pointer rounded-full border-[3px] border-[#757EE4] p-2 text-[#757EE4]"
                @click.prevent="showVideo('{{ $item['youtube'] }}')"
            >
                <x-tabler-player-play class="size-8" />
            </div>
        @endif
        <div @class([
            'mb-6 flex items-center',
            'size-[53px]' => $item['type'] === 'single',
        ])>
            <img
                @class([
                    'w-full h-auto rounded-xl' => $item['type'] === 'bundle',
                ])
                src="{{ $item['icon'] }}"
                @if ($item['type'] === 'single') width="53"
                height="53" @endif
                alt="{{ $item['name'] }}"
            >
            @if ($item['installed'])
                <p class="mb-0 ms-3 flex items-center gap-2 text-2xs font-medium">
                    <span class="inline-block size-2 rounded-full bg-green-500"></span>
                    {{ __('Installed') }}
                </p>
            @endif
        </div>

        <div class="mb-7 flex flex-wrap items-center gap-2">
            <h3 class="m-0 text-xl font-semibold">
                {{ $item['name'] }}
            </h3>
			@if($item['type']!=='bundle')
				<p class="review m-0 flex items-center gap-1 text-sm font-medium text-heading-foreground">
					<x-tabler-star-filled class="size-3" />
					{{ number_format($item['review'], 1) }}
				</p>
			@endif

        </div>
        <p class="mb-7 text-base leading-normal">
            {{ $item['description'] }}
        </p>
        <a
            class="absolute inset-0 z-1"
            href="{{ route('dashboard.admin.marketplace.extension', ['slug' => $item['slug']]) }}"
        >
            <span class="sr-only">
                {{ __('View details') }}
            </span>
        </a>
        <div class="mt-auto flex items-center justify-between">
            @if (!$item['only_show'] && $item['type'] === 'single')
                <div class="flex flex-wrap items-center gap-2">
                    @foreach ($item['categories'] as $tag)
                        {{ $tag }}
                        @if (!$loop->last)
                            <span class="inline-block size-1 rounded-full bg-foreground/10"></span>
                        @endif
                    @endforeach
                </div>
            @else
                <div class="border ps-2 pe-2 p-1 text-white rounded-2xl  border-gray-500 bg-gray-500">@lang('Limited Time Offer')</div>

				<div>
					<span class="text-sm ps-2 pe-2 p-1 text-green-600 rounded-2xl" style="background: #00800024">Save {{ $item['bundle_discount_percent'] }}%</span>
				</div>
            @endif


            @includeWhen($item['type'] === 'single', 'default.panel.admin.marketplace.particles.index-add-cart')
        </div>
    </x-card>
</div>
