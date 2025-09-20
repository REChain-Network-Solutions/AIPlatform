@extends('panel.layout.app', ['disable_tblr' => true])
@section('title', __('Marketplace'))
@section('titlebar_actions_before')
    @php
        $filters = ['All', 'Installed', 'Free', 'Paid'];
    @endphp
    <div
        class="group flex flex-nowrap"
        x-data="{
            searchbarHidden: false
        }"
        :class="searchbarHidden ? 'searchbar-hidden' : ''"
    >
        <x-dropdown.dropdown
            class:dropdown-dropdown="max-lg:end-auto max-lg:start-0 max-sm:-left-20"
            anchor="end"
            triggerType="click"
            offsetY="0px"
        >
            <x-slot:trigger
                class="size-9"
                variant="none"
                title="{{ __('Filter') }}"
            >
                <svg
                    class="flex-shrink-0 cursor-pointer"
                    width="14"
                    height="10"
                    viewBox="0 0 14 10"
                    fill="none"
                    xmlns="http://www.w3.org/2000/svg"
                >
                    <path
                        class="fill-[#0F0F0F] dark:fill-white"
                        d="M5.58333 9.25V7.83333H8.41667V9.25H5.58333ZM2.75 5.70833V4.29167H11.25V5.70833H2.75ZM0.625 2.16667V0.75H13.375V2.16667H0.625Z"
                    />
                </svg>
            </x-slot:trigger>
            <x-slot:dropdown
                class="min-w-28 text-xs font-medium"
            >
                <ul>
                    @foreach ($filters as $filter)
                        <li>
                            <x-button
                                class="lqd-filter-btn addons_filter {{ $loop->first ? 'active' : '' }} w-full justify-start rounded-md text-start transition-colors hover:bg-foreground/5 active:bg-foreground/5"
                                data-filter="{{ $filter }}"
                                type="button"
                                name="filter"
                                variant="none"
                            >
                                {{ __($filter) }}
                            </x-button>
                        </li>
                    @endforeach
                </ul>
            </x-slot:dropdown>
        </x-dropdown.dropdown>
    </div>
@endsection
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
        <div class="flex flex-col gap-9">
            @include('panel.admin.market.components.marketplace-filter')
            {{-- TODO: This banner section should be made in accordance with the design. --}}
            @if (is_array($banners) && $banners)
                <div
                    class="relative flex justify-between overflow-hidden rounded-2xl bg-gradient-to-r from-gradient-via/40 to-gradient-from/40"
                    x-data="{
                        banners: {{ json_encode($banners) }},
                        currentBanner: 0,
                    }"
                >
                    <div class="self-center p-9">
                        <span
                            class="mb-4 inline-block rounded-full bg-heading-foreground/40 px-3 py-1 text-xs/tight font-medium text-background"
                            x-text="banners[currentBanner].banner_title"
                        >
                            {{ $banners[0]['banner_title'] ?? '' }}
                        </span>
                        <h2
                            class="mb-0"
                            x-html="banners[currentBanner].banner_description"
                        >
                            {{ $banners[0]['banner_description'] ?? '' }}
                        </h2>
                        <div class="relative z-10 inline-flex justify-start gap-2 pt-5">
                            @foreach ($banners as $banner)
                                <span
                                    @class([
                                        'relative size-[5px] cursor-pointer rounded-full bg-heading-foreground/10 transition-all before:absolute before:left-1/2 before:top-1/2 before:size-4 before:-translate-x-1/2 before:-translate-y-1/2 hover:bg-heading-foreground/50 [&.active]:w-2.5 [&.active]:bg-heading-foreground',
                                        'active' => $loop->first,
                                    ])
                                    :class="{ 'active': currentBanner === {{ $loop->index }} }"
                                    @click="currentBanner = {{ $loop->index }}"
                                ></span>
                            @endforeach
                        </div>
                    </div>

                    <div class="self-center">
                        <img
                            class="h-32 object-cover"
                            src="{{ $banners[0]['banner_image'] ?? '' }}"
                            alt="{{ $banners[0]['banner_title'] ?? '' }}"
                            :src="banners[currentBanner].banner_image"
                            :alt="banners[currentBanner].banner_title"
                        >
                    </div>

                    <a
                        class="absolute inset-0 z-1 inline-block"
                        :href="banners[currentBanner].banner_link"
                        href="{{ $banners[0]['banner_link'] ?? '' }}"
                    ></a>
                </div>
            @endif

            <x-alerts.payment-status :payment-status="$paymentStatus" />
            <div class="lqd-extension-grid grid grid-cols-1 gap-7 md:grid-cols-2 lg:grid-cols-3">

                @foreach ($items as $item)
                    {{-- TODO: {{ $item['is_featured'] ? 'border-red-500': '' --- If is featured true, a border gradient must be added. --}}
                   	@include('default.panel.admin.marketplace.particles.index-item')
                @endforeach
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

@include('panel.admin.marketplace.script')
