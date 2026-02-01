<div
    class="lqd-adv-img-editor-gallery pointer-events-none invisible fixed inset-0 z-2 flex min-h-screen overflow-y-auto bg-background opacity-0 transition-all"
    :class="{
        'opacity-0': currentView !== 'gallery',
        'invisible': currentView !== 'gallery',
        'pointer-events-none': currentView !== 'gallery'
    }"
    x-data="{
        cols: 1,
        init() {
            this.cols = this.getCols();
        },
        increaseCols() {
            this.cols = Math.min(6, this.cols + 1);
        },
        decreaseCols() {
            this.cols = Math.max(0, this.cols - 1);
        },
        getCols() {
            const imageGrid = this.$refs.galleryImageGrid;
            const gridStyles = window.getComputedStyle(imageGrid);
    
            return parseInt(gridStyles.getPropertyValue('--cols'));
        }
    }"
>
    <div class="container">
        <div class="py-28">
            <div class="mb-10 flex flex-wrap items-center justify-between gap-x-2 gap-y-4">
                <h2 class="m-0">
                    @lang('Gallery')
                </h2>

                <div class="flex flex-wrap items-center gap-3 text-label lg:flex-nowrap">
                    <label
                        class="text-2xs font-medium text-heading-foreground/80"
                        for="gallery_columns"
                    >
                        @lang('Columns')
                    </label>
                    <div class="flex w-full max-w-60 items-center gap-3">
                        <button
                            class="inline-grid size-4 place-content-center"
                            type="button"
                            @click.prevent="decreaseCols"
                        >
                            <x-tabler-minus class="size-4" />
                        </button>
                        <input
                            class="h-0.5 w-full appearance-none rounded-full bg-neutral-50 focus:outline-black dark:bg-neutral-900 dark:focus:outline-white [&::-moz-range-thumb]:size-2.5 [&::-moz-range-thumb]:appearance-none [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:border [&::-moz-range-thumb]:border-none [&::-moz-range-thumb]:border-background [&::-moz-range-thumb]:bg-black active:[&::-moz-range-thumb]:scale-110 [&::-moz-range-thumb]:dark:bg-white [&::-webkit-slider-thumb]:size-2.5 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:border [&::-webkit-slider-thumb]:border-none [&::-webkit-slider-thumb]:border-background [&::-webkit-slider-thumb]:bg-black active:[&::-webkit-slider-thumb]:scale-110 [&::-webkit-slider-thumb]:dark:bg-white"
                            id="gallery_columns"
                            type="range"
                            value="1"
                            min="1"
                            max="6"
                            step="1"
                            x-model="cols"
                        />
                        <button
                            class="inline-grid size-4 place-content-center"
                            type="button"
                            @click.prevent="increaseCols"
                        >
                            <x-tabler-plus class="size-4" />
                        </button>
                    </div>
                </div>
            </div>

            <div id="lqd-realtime-image-recent-images-grid-wrap">
                <div
                    class="lqd-realtime-image-recent-images-grid grid grid-cols-[repeat(var(--cols),minmax(0,1fr))] gap-5 transition-all [--cols:1] sm:[--cols:2] md:gap-x-6 md:[--cols:3] lg:gap-x-11 lg:[--cols:5] [&_.image-result:nth-child(n+17)]:hidden"
                    x-ref="galleryImageGrid"
                    :style="{ '--cols': cols }"
                >
                    @include('ai-realtime-image::shared-components.image-grid-items', ['images' => $images, 'id_prefix' => 'gallery-'])
                </div>

                <div class="mt-4">
                    {{ $images->links('pagination::ajax', [
                        'action' => route('dashboard.user.ai-realtime-image.gallery'),
                        'target_id' => 'lqd-realtime-image-recent-images-grid-wrap',
                        'ajax_after' => '$nextTick(() => document.querySelectorAll(".lqd-realtime-image-new[data-id-prefix=\"gallery-\"]").forEach(el => el.remove()))',
                    ]) }}
                </div>
            </div>
        </div>
    </div>
</div>
