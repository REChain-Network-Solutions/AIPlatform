@php
    $price_details = [
        [
            'label' => 'License',
            'value' => 'Lifetime',
        ],
        [
            'label' => 'Support & Updates',
            'value' => '6 months',
        ],
        [
            'label' => 'Installation',
            'value' => 'One Click',
        ],
        [
            'label' => 'Recently Updated',
            'value' => 'Yes',
        ],
    ];
@endphp

@extends('panel.layout.app', ['disable_tblr' => true])
@section('title', __('Marketplace'))
@section('titlebar_pretitle')
    <x-button
        class="text-inherit hover:text-foreground"
        variant="link"
        href="{{ route('dashboard.admin.marketplace.index') }}"
    >
        <x-tabler-chevron-left
            class="size-4"
            stroke-width="1.5"
        />
        {{ __('Back to Marketplace') }}
    </x-button>
@endsection

@push('before-head-close')
    <style>
        .extension-detail ul {
            list-style: disc;
            margin-left: 0.8rem;
        }
    </style>
@endpush
@section('titlebar_actions')
    <div class="flex flex-wrap gap-2">
        <x-button
            variant="ghost-shadow"
            href="{{ route('dashboard.admin.marketplace.liextension') }}"
        >
            {{ __('Manage Addons') }}
        </x-button>
        <x-button href="{{ route('dashboard.admin.marketplace.index') }}">
            <x-tabler-plus class="size-4" />
            {{ __('Browse Add-ons') }}
        </x-button>
        <x-button
            class="relative ms-2"
            variant="ghost-shadow"
            href="{{ route('dashboard.admin.marketplace.cart') }}"
        >
            <x-tabler-shopping-cart class="size-4" />
            {{ __('Cart') }}
            <small
                class="absolute right-[3px] top-[-10px] rounded-[50%] border border-red-500 bg-red-500 pe-2 ps-2 text-white"
                id="itemCount"
            >{{ count(is_array($cart) ? $cart : []) }}</small>
        </x-button>
    </div>
@endsection

@section('content')
    <div
        class="py-10"
        x-data="{
            open: false,
            youtubeId: null,
            showVideo(id) {
                this.youtubeId = id;
                this.open = true;
            },
            closeVideo() {
                this.youtubeId = null;
                this.open = false;
            }
        }"
    >
        <div class="lqd-extension-details flex flex-col justify-between gap-y-7 md:flex-row">
            <x-card
                class="lqd-extension-details-card relative w-full max-w-none pb-10 lg:w-8/12 [&_hr]:my-5 [&_hr]:border-border"
                variant="shadow"
                size="lg"
            >
                <img
                    class="mb-8 h-auto w-full"
                    src="{{ $item['icon'] }}"
                >

                @if ($item['youtube'])
                    <div
                        class="absolute end-10 top-10 z-[1000000000000000] cursor-pointer rounded-full border-[3px] border-[#757EE4] p-2 text-[#757EE4]"
                        @click.prevent="showVideo('{{ $item['youtube'] }}')"
                    >
                        <x-tabler-player-play class="size-12" />
                    </div>
                @endif

                <div class="mb-8 flex flex-wrap items-center gap-2">
                    <h3 class="m-0 text-[23px] font-semibold">
                        {{ $item['name'] }}
                    </h3>

{{--                    @if ($item['installed'] and $item['version'] == $item['db_version'])--}}
{{--                        <p class="mb-0 ms-3 flex items-center gap-2 text-2xs font-medium">--}}
{{--                            <span class="inline-block size-2 rounded-full bg-green-500"></span>--}}
{{--                            {{ __('Installed') }}--}}
{{--                        </p>--}}
{{--                    @else--}}
{{--                        <a class="mb-0 ms-3 flex items-center gap-2 text-2xs font-medium">--}}
{{--                            <span class="inline-block size-2 rounded-full bg-green-500"></span>--}}
{{--                            {{ __('Upgrade') }}--}}
{{--                        </a>--}}
{{--                    @endif--}}
                </div>

                <h3 class="mb-7">
                    {{ __('About this bundle') }}
                </h3>

                <div class="extension-detail prose prose-sm mb-8 max-w-none dark:prose-invert">
                    {!! $item['detail'] !!}
                </div>

                <div class="mb-11 flex flex-col gap-4">
                    @foreach ($item['categories'] as $tag)
                        <p class="flex items-center gap-3 text-base font-medium">
                            <span
                                class="me-1 inline-flex size-6 items-center justify-center rounded-xl bg-primary/[8%] align-middle text-primary dark:bg-secondary/15 dark:text-secondary-foreground"
                            >
                                <x-tabler-check class="size-3.5" />
                            </span>
                            {{ $tag }}
                        </p>
                    @endforeach
                </div>

                <div>
                    <div class="mb-7 flex flex-wrap items-center justify-between gap-3">
                        <h3 class="mb-0">
                            {{ __('Addons Included In This Bundle ✅') }}
                        </h3>

                        <span class="inline-flex items-center rounded-full bg-emerald-500/15 px-4 py-1.5 text-sm font-semibold text-emerald-600">
                            {{ trans('Bundle Deal: Save') . ' ' . $item['bundle_discount_percent'] . '%' }}
                            <svg
                                class="ms-1"
                                width="17"
                                height="17"
                                viewBox="0 0 17 17"
                                fill="currentColor"
                                xmlns="http://www.w3.org/2000/svg"
                            >
                                <path
                                    d="M7.75067 0.583344C8.38051 0.583478 8.9845 0.833789 9.42979 1.27922L15.5335 7.38297C16.0397 7.88922 16.324 8.57579 16.324 9.29168C16.324 10.0076 16.0397 10.6941 15.5335 11.2004L11.1065 15.6274C10.6003 16.1335 9.91372 16.4179 9.19784 16.4179C8.48195 16.4179 7.79538 16.1335 7.28913 15.6274L1.18538 9.52364C0.739947 9.07834 0.489636 8.47435 0.489502 7.84451V3.75001C0.489502 2.91016 0.823132 2.1047 1.417 1.51084C2.01086 0.916974 2.81632 0.583344 3.65617 0.583344H7.75067ZM4.84367 3.35418C4.44421 3.35405 4.05947 3.50492 3.76657 3.77653C3.47367 4.04814 3.29425 4.42043 3.26429 4.81876L3.26034 4.93751C3.26034 5.25066 3.3532 5.55679 3.52718 5.81716C3.70115 6.07754 3.94844 6.28048 4.23775 6.40032C4.52707 6.52016 4.84543 6.55151 5.15256 6.49042C5.4597 6.42933 5.74182 6.27853 5.96325 6.0571C6.18469 5.83566 6.33549 5.55354 6.39658 5.2464C6.45767 4.93927 6.42632 4.62091 6.30648 4.33159C6.18664 4.04228 5.9837 3.795 5.72332 3.62102C5.46294 3.44704 5.15682 3.35418 4.84367 3.35418Z"
                                />
                            </svg>
                        </span>
                    </div>

                    <div class="mb-6 space-y-6">
                        @php
                            $totalProductPrice = 0;
                        @endphp
                        @foreach ($item['bundleExtensions'] as $bundleItem)
                            <x-card
                                class:body="flex gap-6"
                                class="text-base/6 hover:-translate-y-1 hover:shadow-lg hover:shadow-black/5"
                            >
                                <figure class="w-11 shrink-0">
                                    <img
                                        class="h-auto w-full"
                                        src="{{ $bundleItem['icon'] }}"
                                        alt="{{ $bundleItem['name'] }}"
                                    >
                                </figure>

                                <div class="grow">
                                    <div class="mb-2.5 flex items-center gap-2">
                                        <h3 class="mb-0">
                                            {{ $bundleItem['name'] }}
                                        </h3>
                                        <span class="inline-flex items-center gap-1 text-heading-foreground">
                                            <x-tabler-star-filled class="size-3" />
                                            {{ number_format($bundleItem['review'], 1) }}
                                        </span>
                                    </div>
                                    <p class="mb-3">
                                        {{ $bundleItem['description'] }}
                                    </p>
                                    <a
                                        class="underline underline-offset-2"
                                        target="_blank"
                                        href="{{ route('dashboard.admin.marketplace.extension', ['slug' => $bundleItem['slug']]) }}"
                                    >
                                        @lang('Learn more')
                                    </a>
                                </div>
                                <a
                                    class="absolute inset-0"
                                    target="_blank"
                                    href="{{ route('dashboard.admin.marketplace.extension', ['slug' => $bundleItem['slug']]) }}"
                                ></a>
                            </x-card>

                            @php
                                $totalProductPrice += $bundleItem['price'];
                            @endphp
                        @endforeach
                    </div>

                    <div class="mb-5">
                        <div class="flex justify-between gap-1 border-b py-3 last:border-b-0">
                            <span class="opacity-60">
                                {{ __('Price of Individual Addons') }}
                            </span>
                            <span class="text-heading-foreground">
                                ${{ $totalProductPrice }}
                            </span>
                        </div>
                        <div class="flex justify-between gap-1 border-b py-3 last:border-b-0">
                            <span class="opacity-60">
                                {{ __('Bundle Save') }}
                            </span>
                            <span class="text-green-600">
                                ${{ $totalProductPrice -$item['price'] }}
                            </span>
                        </div>
                        <div class="flex justify-between gap-1 border-b py-3 text-[18px] font-semibold last:border-b-0">
                            <span class="opacity-60">
                                {{ __('Total') }}
                            </span>
                            <div class="font-bold text-heading-foreground">
{{--                                @if ($item['fake_price'] ?? false)--}}
{{--                                    <s class="text-[18px] line-through">--}}
{{--                                        ${{ $item['fake_price'] }}--}}
{{--                                    </s>--}}
{{--                                @endif--}}
                                ${{ $item['price'] }}
                            </div>
                        </div>
                    </div>

                    <x-button
                        class="w-full shadow-xl shadow-black/10"
                        variant="success"
                        target="_blank"
                        size="lg"
                        href="{{ $item['routes']['payment'] }}"
                    >
                        {{ __('Purchase Bundle') }}
                    </x-button>
                </div>
            </x-card>

            <div class="flex w-full flex-col gap-8 lg:w-4/12 lg:ps-8">
                <x-card
                    class="lqd-extension-price-card text-center"
                    size="lg"
                >
                    <span class="mb-4 inline-flex items-center rounded-full bg-emerald-500/15 px-4 py-1.5 text-sm font-semibold text-emerald-600">
                        {{ trans('Bundle Deal: Save') . ' ' . $item['bundle_discount_percent'] . '%' }}
                        <svg
                            class="ms-1"
                            width="17"
                            height="17"
                            viewBox="0 0 17 17"
                            fill="currentColor"
                            xmlns="http://www.w3.org/2000/svg"
                        >
                            <path
                                d="M7.75067 0.583344C8.38051 0.583478 8.9845 0.833789 9.42979 1.27922L15.5335 7.38297C16.0397 7.88922 16.324 8.57579 16.324 9.29168C16.324 10.0076 16.0397 10.6941 15.5335 11.2004L11.1065 15.6274C10.6003 16.1335 9.91372 16.4179 9.19784 16.4179C8.48195 16.4179 7.79538 16.1335 7.28913 15.6274L1.18538 9.52364C0.739947 9.07834 0.489636 8.47435 0.489502 7.84451V3.75001C0.489502 2.91016 0.823132 2.1047 1.417 1.51084C2.01086 0.916974 2.81632 0.583344 3.65617 0.583344H7.75067ZM4.84367 3.35418C4.44421 3.35405 4.05947 3.50492 3.76657 3.77653C3.47367 4.04814 3.29425 4.42043 3.26429 4.81876L3.26034 4.93751C3.26034 5.25066 3.3532 5.55679 3.52718 5.81716C3.70115 6.07754 3.94844 6.28048 4.23775 6.40032C4.52707 6.52016 4.84543 6.55151 5.15256 6.49042C5.4597 6.42933 5.74182 6.27853 5.96325 6.0571C6.18469 5.83566 6.33549 5.55354 6.39658 5.2464C6.45767 4.93927 6.42632 4.62091 6.30648 4.33159C6.18664 4.04228 5.9837 3.795 5.72332 3.62102C5.46294 3.44704 5.15682 3.35418 4.84367 3.35418Z"
                            />
                        </svg>
                    </span>

                    @if ($item['price'] != 0 && !$item['only_show'])
                        <div class="mb-5">
                            <p class="text-4xl/none font-semibold text-heading-foreground">
                                @if (currencyShouldDisplayOnRight(currency()['symbol']))
                                    @if ($item['fake_price'] ?? false)
                                        <s class="text-[0.64em] opacity-50">
                                            {{ $item['fake_price'] }}$
                                        </s>
                                    @endif
                                    {{ $item['price'] }}$
                                @else
                                    @if ($item['fake_price'] ?? false)
                                        <s class="text-[0.64em] opacity-50">
                                            ${{ $item['fake_price'] }}
                                        </s>
                                    @endif
                                    ${{ $item['price'] }}
                                @endif
                            </p>
                            <p class="m-0 text-xl font-semibold text-green-600">
                                {{ __('Limited Time Offer') }}
                            </p>
                        </div>

                        <p class="mb-6 text-2xs opacity-60">
                            {{ __('Price is in US dollars. Tax included.') }}
                        </p>
                    @else
                        <p class="mb-5 text-4xl font-semibold leading-none text-heading-foreground">
                            @if (!$item['only_show'])
                                {{ __('Free') }}
                            @else
                                {{ __('Contact Us') }}
                            @endif
                        </p>
                    @endif

                    @if (!$item['only_show'])
                        <div class="justify-{{ $item['is_buy'] ? 'between' : 'center' }} flex gap-2">
                            @if ($item['price'] != 0)
                                @if ($app_is_demo)
                                    <x-button
                                        class="w-full shadow-xl shadow-black/10"
                                        variant="success"
                                        size="lg"
                                        href="#"
                                        onclick="return toastr.info('This feature is disabled in Demo version.')"
                                        disabled
                                    >
                                        {{ __('Purchase Bundle') }}
                                    </x-button>
                                @else
                                    @php
                                        $is_license = $item['licensed'] == 1;
                                    @endphp

                                    @if ($is_license)
                                        @if ($item['support']['support'] === false && $item['installed'])
                                            <div>
                                                <x-button
                                                    class="mb-3 w-full"
                                                    size="lg"
                                                    href="{{ $item['routes']['paymentSupport'] }}"
                                                >
                                                    {{ __('Renew License') }}
                                                </x-button>

                                                <x-modal
                                                    class:modal-backdrop="backdrop-blur-none bg-foreground/15"
                                                    class="inline-flex"
                                                    title="{{ __('Your update and support period has ended.') }}"
                                                >
                                                    <x-slot:trigger
                                                        size="none"
                                                        variant="ghost-shadow"
                                                        @class([
                                                            'p-1 ps-4 pe-4 text-red-500 hover:bg-red-300 bg-red-100 w-full',
                                                        ])
                                                    >
                                                        @lang('Update & Support period has expired.')
                                                    </x-slot:trigger>

                                                    <x-slot:modal>

                                                        <p>
                                                            Your extension license remains active, but access to new updates <br> and support ended after the initial 6-month period.
                                                            <span class="underline">Extend your <br> license period to get the latest features, updates, and dedicated <br>
                                                                support.</span>
                                                        </p>

                                                        <p class="mt-4">Alternatively, you can continue using your current extension<br> version, but without access to new features
                                                            or support.</p>

                                                        <x-button
                                                            class="mt-3 w-full text-2xs font-semibold"
                                                            variant="secondary"
                                                            href="{{ $item['routes']['paymentSupport'] }}"
                                                        >
                                                            @lang('Renew License')
                                                            <span
                                                                class="inline-grid size-7 place-items-center rounded-full bg-background text-heading-foreground shadow-xl"
                                                                aria-hidden="true"
                                                            >
                                                                <x-tabler-chevron-right class="size-4" />
                                                            </span>
                                                        </x-button>

                                                    </x-slot:modal>
                                                </x-modal>

                                            </div>
                                        @else
                                            <x-button
                                                class="w-full"
                                                size="lg"
                                                href="{{ route('dashboard.admin.marketplace.liextension') }}"
                                            >
                                                {{ __('Install Now') }}
                                            </x-button>
                                        @endif
                                    @else
                                        @include('default.panel.admin.marketplace.particles.bundle-is-buy')
                                    @endif

                                @endif
                            @else
                                @if ($item['installed'])
                                    @if ($item['version'] == $item['db_version'])
                                        <x-button
                                            class="w-full"
                                            size="lg"
                                            href="#"
                                            variant="outline"
                                        >
                                            {{ __('Installed') }}
                                        </x-button>
                                    @else
                                        <x-button
                                            class="w-full"
                                            size="lg"
                                            href="{{ route('dashboard.admin.marketplace.liextension') }}"
                                        >
                                            {{ __('Upgrade') }}
                                        </x-button>
                                    @endif
                                @else
                                    <x-button
                                        class="w-full"
                                        size="lg"
                                        href="{{ route('dashboard.admin.marketplace.liextension') }}"
                                    >
                                        {{ __('Install Now') }}
                                    </x-button>
                                @endif

                            @endif
                        </div>
                    @else
                        <x-button
                            class="w-full"
                            size="lg"
                            target="_blank"
                            href="{!! $item['routes']['redirect'] ? $item['routes']['redirect'] . '&email=' . auth()->user()->email() : 'https://mobile.magicproject.ai' !!} "
                        >
                            {{ __('Live Preview') }}
                        </x-button>
                    @endif

                </x-card>

                @if (!$item['only_show'])
                    <x-card
                        class="lqd-extension-price-details"
                        size="lg"
                    >
                        <h4 class="mb-6 border-b pb-3">
                            {{ __('Details') }}
                        </h4>
                        <div class="grid grid-cols-2 gap-4">
                            @foreach ($price_details as $detail)
                                <div class="lqd-extension-price-detail flex flex-col rounded-xl border px-4 py-3 font-semibold">
                                    <p class="mb-6 text-4xs uppercase tracking-widest text-heading-foreground opacity-70">
                                        {{ __($detail['label']) }}
                                    </p>
                                    <p class="mt-auto text-heading-foreground">
                                        {{ __($detail['value']) }}
                                    </p>
                                </div>
                            @endforeach
                        </div>
                    </x-card>
                @endif

            </div>

        </div>
        <div
            class="fixed inset-0 z-[9999999999] flex items-center justify-center bg-black bg-opacity-80"
            id="youtubeModal"
            x-show="open"
            x-transition
            x-cloak
            @keydown.escape.window="closeVideo()"
            @click.outside="closeVideo()"
            style="display: none"
        >
            <button
                class="absolute right-4 top-3 z-20 text-3xl font-bold text-white"
                @click="closeVideo()"
            >×</button>
            <div
                class="relative aspect-video w-[90vw] max-w-4xl overflow-hidden rounded-xl bg-black shadow-lg"
                @click.outside="closeVideo()"
            >
                <iframe
                    class="h-full w-full"
                    :src="youtubeId ? `https://www.youtube.com/embed/${youtubeId}?autoplay=1` : ''"
                    title="YouTube video"
                    frameborder="0"
                    allow="autoplay; encrypted-media"
                    allowfullscreen
                ></iframe>
            </div>
        </div>
    </div>
@endsection

@push('script')
    <script src="{{ custom_theme_url('/assets/js/panel/marketplace.js') }}"></script>
    <script>
        document.getElementById('copyButton')?.addEventListener('click', function() {
            navigator.clipboard.writeText('info@liquid-themes.com');
            toastr.success(@json(__('Copied')));
        });
    </script>
@endpush
