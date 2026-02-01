<nav
    class="lqd-realtime-image-nav fixed inset-x-0 top-0 z-10 flex min-h-[--header-h] gap-6 border-b border-heading-foreground/5 bg-background/10 px-6 py-2.5 backdrop-blur-xl backdrop-saturate-[120%]">
    <div class="flex grow items-center gap-6">
        <div
            class="inline-grid size-[36px] place-content-center overflow-hidden transition-all duration-300"
            :class="{ 'w-0': currentView !== 'home', 'size-[36px]': currentView === 'home', '-me-6': currentView !== 'home' }"
        >
            <x-button
                class="size-[34px] hover:translate-y-0"
                variant="outline"
                hover-variant="primary"
                size="none"
                title="{{ __('Dashboard') }}"
                href="{{ route('dashboard.user.index') }}"
                {{-- ::class="{ 'hidden': currentView !== 'home' }" --}}
            >
                <x-tabler-chevron-left class="size-4" />
            </x-button>
        </div>

        <x-header-logo />

        <div class="hidden items-center gap-6 text-2xs">
            <span>
                @lang('Remaining Credits'):
                <b>
                    {{ (int) Auth::user()->remaining_images != -1 ? rtrim(rtrim(number_format((float) Auth::user()->remaining_images, 2), '0'), '.') : __('Unlimited') }}
                </b>
            </span>
            <span class="relative h-[5px] w-[150px] overflow-hidden rounded-full bg-heading-foreground/5">
                {{-- @TODO: calculate the width based on total and remaining images --}}
                <span
                    class="absolute start-0 top-0 inline-block h-full bg-primary"
                    style="width: 100%"
                ></span>
            </span>
        </div>
    </div>

    <div class="flex select-none items-center justify-end gap-6">
        <x-button
            class="gap-2.5 px-5 py-2.5"
            variant="outline"
            hover-variant="primary"
            ::class="{ 'hidden': currentView !== 'home' }"
            @click.prevent="switchView('gallery')"
        >
            @lang('View Gallery')
            <svg
                width="17"
                height="12"
                viewBox="0 0 17 12"
                fill="currentColor"
                xmlns="http://www.w3.org/2000/svg"
            >
                <path
                    d="M1.41948 11.8625C0.977456 11.8625 0.632487 11.7385 0.38457 11.4906C0.136654 11.2427 0.0126953 10.8977 0.0126953 10.4557V1.54428C0.0126953 1.10226 0.136654 0.757289 0.38457 0.509372C0.632487 0.261455 0.977456 0.137497 1.41948 0.137497H3.43526C3.87728 0.137497 4.22225 0.261455 4.47016 0.509372C4.71808 0.757289 4.84204 1.10226 4.84204 1.54428V10.4557C4.84204 10.8977 4.71808 11.2427 4.47016 11.4906C4.22225 11.7385 3.87728 11.8625 3.43526 11.8625H1.41948ZM1.41948 10.725H3.43526C3.51386 10.725 3.57839 10.6998 3.62885 10.6493C3.67931 10.5989 3.70454 10.5343 3.70454 10.4557V1.54428C3.70454 1.46567 3.67931 1.40114 3.62885 1.35069C3.57839 1.30023 3.51386 1.275 3.43526 1.275H1.41948C1.34087 1.275 1.27634 1.30023 1.22588 1.35069C1.17542 1.40114 1.1502 1.46567 1.1502 1.54428V10.4557C1.1502 10.5343 1.17542 10.5989 1.22588 10.6493C1.27634 10.6998 1.34087 10.725 1.41948 10.725ZM7.99204 11.8625C7.55002 11.8625 7.20505 11.7385 6.95713 11.4906C6.70922 11.2427 6.58526 10.8977 6.58526 10.4557V1.54428C6.58526 1.10226 6.70922 0.757289 6.95713 0.509372C7.20505 0.261455 7.55002 0.137497 7.99204 0.137497H15.5809C16.0229 0.137497 16.3679 0.261455 16.6158 0.509372C16.8637 0.757289 16.9877 1.10226 16.9877 1.54428V10.4557C16.9877 10.8977 16.8637 11.2427 16.6158 11.4906C16.3679 11.7385 16.0229 11.8625 15.5809 11.8625H7.99204ZM7.99204 10.725H15.5809C15.6595 10.725 15.7241 10.6998 15.7745 10.6493C15.825 10.5989 15.8502 10.5343 15.8502 10.4557V1.54428C15.8502 1.46567 15.825 1.40114 15.7745 1.35069C15.7241 1.30023 15.6595 1.275 15.5809 1.275H7.99204C7.91344 1.275 7.84891 1.30023 7.79845 1.35069C7.74799 1.40114 7.72276 1.46567 7.72276 1.54428V10.4557C7.72276 10.5343 7.74799 10.5989 7.79845 10.6493C7.84891 10.6998 7.91344 10.725 7.99204 10.725Z"
                />
            </svg>
        </x-button>
    </div>

    <x-button
        class="hidden size-[34px] shrink-0"
        variant="outline"
        hover-variant="primary"
        size="none"
        title="{{ __('Close Editor') }}"
        ::class="{ 'hidden': currentView === 'home', 'inline-flex': currentView !== 'home' }"
        @click.prevent="switchView('home')"
    >
        <x-tabler-x class="size-4" />
    </x-button>
</nav>
