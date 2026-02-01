<div
    class="lqd-realtime-image-home transition-all"
    :class="{
        'opacity-0': currentView !== 'home',
        'invisible': currentView !== 'home',
        'pointer-events-none': currentView !== 'home'
    }"
>
    <div class="container">
        @include('ai-realtime-image::home.generator-form', ['image_styles' => $image_styles])

        <div
            class="mx-auto py-3 lg:w-9/12"
            x-cloak
            x-show="newImages.length || requestSent"
            x-transition
        >
            <figure
                class="rounded-lg bg-heading-foreground/5"
                :class="{ 'motion-opacity-loop-75': busy }"
            >
                <img
                    class="h-auto w-full rounded-lg"
                    src="data:image/svg+xml;charset=utf-8,%3Csvg xmlns%3D'http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg' viewBox%3D'0 0 1024 758'%2F%3E"
                    :src="newImages.at(0)?.image || $el.getAttribute('src')"
                    :alt="newImages.at(0)?.prompt || ''"
                    width="1024"
                    height="758"
                    loading="lazy"
                >
            </figure>
        </div>

        @includeWhen(!$app_is_demo, 'ai-realtime-image::home.recent-images-grid')
    </div>
</div>
