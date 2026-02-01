<div
    {{ $attributes->withoutTwMergeClasses()->twMerge('lqd-sidedrawer pointer-events-none') }}
    x-data="lqdSidedrawer"
    :class="{ 'pointer-events-none': !sidedrawerOpen }"
>
    <div
        {{ $attributes->twMergeFor('backdrop', 'lqd-sidedrawer-backdrop fixed start-0 top-0 z-[99] h-screen w-screen bg-black/50') }}
        x-ref="sidedrawerBackdrop"
        x-show="sidedrawerOpen"
        x-transition.opacity
        x-cloak
        @click.prevent="sidedrawerOpen = false"
    ></div>
    <div
        {{ $attributes->twMergeFor('content-wrap', 'lqd-sidedrawer-content-wrap fixed bottom-0 end-0 top-0 z-[100] w-[min(420px,85vw)] translate-x-full overflow-hidden border-s bg-background transition duration-300 ease-out overscroll-contain') }}
        :class="{ 'translate-x-0 shadow-lg shadow-black/5': sidedrawerOpen, 'translate-x-full shadow-none': !sidedrawerOpen }"
    >
        <div {{ $attributes->twMergeFor('content', 'lqd-sidedrawer-content h-full w-full overflow-y-auto overscroll-contain') }}>
            {{ $slot }}
        </div>
    </div>
</div>
