<nav
    class="lqd-adv-img-editor-nav fixed inset-x-0 top-0 z-10 flex min-h-[--header-h] gap-6 border-b border-heading-foreground/5 bg-background/10 px-6 py-2.5 backdrop-blur-xl backdrop-saturate-[120%]">
    <div class="flex grow items-center gap-6">
        <div
            class="size-[36px] inline-grid place-content-center overflow-hidden transition-all duration-300"
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

        <div
            class="flex items-center gap-6"
            :class="{ 'opacity-50': busy, 'pointer-events-none': busy }"
            x-cloak
            x-show="currentView === 'editor' && (editingImage.output || selectedTool === 'sketch_to_image')"
            x-transition:enter="transition ease duration-200"
            x-transition:enter-start="opacity-0 translate-x-3"
            x-transition:enter-end="opacity-100 translate-x-0"
            x-transition:leave="transition ease duration-200"
            x-transition:leave-start="opacity-100 translate-x-0"
            x-transition:leave-end="opacity-0 -translate-x-3"
        >
            <span
                class="inline-block h-6 w-px bg-heading-foreground/[8%]"
                x-show="selectedTool !== 'sketch_to_image'"
            ></span>

            <x-button
                class="shrink-0 text-2xs"
                x-show="selectedTool !== 'sketch_to_image'"
                variant="link"
                @click.prevent="resetUploadedImageInput"
            >
                @lang('New Image')
                <span class="size-[34px] inline-grid place-content-center rounded-full border">
                    <x-tabler-refresh class="size-4 group-hover:animate-spin group-hover:[animation-iteration-count:1]" />
                </span>
            </x-button>

            <div
                class="flex items-center gap-3"
                x-show="selectedTool === 'cleanup' || selectedTool === 'inpainting' || selectedTool === 'sketch_to_image'"
                x-transition
            >
                <span class="inline-block h-6 w-px bg-heading-foreground/[8%]"></span>

                <x-button
                    class="text-2xs"
                    variant="link"
                    @click.prevent="makeCanvasEditable(selectedTool === 'sketch_to_image' ? { width: 1024, height: 1024 } : {})"
                >
                    @lang('Reset Paint')
                    <span class="size-[34px] inline-grid place-content-center rounded-full border">
                        <x-tabler-paint class="size-4" />
                    </span>
                </x-button>
            </div>

            <span
                class="inline-block h-6 w-px bg-heading-foreground/[8%]"
                x-show="selectedTool === 'cleanup' || selectedTool === 'inpainting' || selectedTool === 'sketch_to_image'"
                x-transition
            ></span>

            <div
                class="flex items-center gap-2 text-2xs"
                x-show="selectedTool === 'cleanup' || selectedTool === 'inpainting' || selectedTool === 'sketch_to_image'"
                x-transition
            >
                <span class="text-nowrap">
                    @lang('Brush Size')
                </span>
                <button
                    class="size-4 inline-grid place-content-center"
                    type="button"
                    @click.prevent="setBrushSize('-')"
                >
                    <x-tabler-minus class="size-4" />
                </button>
                <input
                    class="w-8 bg-background appearance-none text-center [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none"
                    type="number"
                    min="10"
                    max="100"
                    @input="setBrushSize($event.target.value)"
                    :value="brushSize"
                    step="10"
                >
                <button
                    class="size-4 inline-grid place-content-center"
                    type="button"
                    @click.prevent="setBrushSize('+')"
                >
                    <x-tabler-plus class="size-4" />
                </button>
            </div>
        </div>
    </div>

    <div
        class="flex select-none items-center justify-end gap-6"
        :class="{ 'opacity-50': busy, 'pointer-events-none': busy }"
    >
        <div
            class="flex items-center gap-4"
            x-cloak
            x-show="currentView === 'editor' && (editingImage.output || selectedTool === 'sketch_to_image')"
            x-transition:enter="transition ease duration-200"
            x-transition:enter-start="opacity-0 translate-x-3"
            x-transition:enter-end="opacity-100 translate-x-0"
            x-transition:leave="transition ease duration-200"
            x-transition:leave-start="opacity-100 translate-x-0"
            x-transition:leave-end="opacity-0 -translate-x-3"
        >
            <x-button
                class="text-2xs"
                variant="link"
                @click.prevent="fitToScreen"
            >
                @lang('Fit to Screen')
            </x-button>

            <div class="flex gap-4">
                <button
                    class="size-5 inline-grid place-content-center rounded-full bg-heading-foreground/5 text-heading-foreground transition-all hover:scale-110 hover:bg-heading-foreground hover:text-heading-background disabled:pointer-events-none disabled:opacity-50"
                    type="button"
                    @click.prevent="zoomOut"
                    :disabled="reachedMinZoom"
                >
                    <x-tabler-minus class="size-3" />
                </button>
                <span class="text-2xs font-medium">
                    <input
                        class="w-6 bg-background"
                        :value="(zoomLevel * 100).toFixed(0)"
                        type="text"
                        @input="setZoomLevel($event.target.value / 100)"
                    />
                    %
                </span>
                <button
                    class="size-5 inline-grid place-content-center rounded-full bg-heading-foreground/5 text-heading-foreground transition-all hover:scale-110 hover:bg-heading-foreground hover:text-heading-background disabled:pointer-events-none disabled:opacity-50"
                    type="button"
                    @click.prevent="zoomIn"
                    :disabled="reachedMaxZoom"
                >
                    <x-tabler-plus class="size-3" />
                </button>
            </div>

            <x-button
                class="hidden text-2xs"
                variant="link"
            >
                @lang('Generate a Variant')
                <span class="size-[34px] inline-grid place-content-center rounded-full border">
                    <svg
                        width="14"
                        height="16"
                        viewBox="0 0 14 16"
                        fill="currentColor"
                        xmlns="http://www.w3.org/2000/svg"
                    >
                        <path
                            d="M6.16665 15.0256C4.60362 14.8226 3.31009 14.134 2.28606 12.9599C1.26202 11.7858 0.75 10.4102 0.75 8.83332C0.75 7.9861 0.917201 7.1744 1.2516 6.39824C1.58601 5.62207 2.05449 4.94018 2.65704 4.35257L3.54804 5.24359C3.03629 5.71047 2.65035 6.2524 2.39021 6.86938C2.13006 7.48637 1.99998 8.14102 1.99998 8.83332C1.99998 10.0555 2.3902 11.1327 3.17065 12.0649C3.9511 12.997 4.94977 13.5673 6.16665 13.7756V15.0256ZM7.83331 15.0416V13.7917C9.0363 13.5481 10.0315 12.965 10.8189 12.0424C11.6063 11.1199 12 10.0502 12 8.83332C12 7.44443 11.5139 6.26388 10.5416 5.29165C9.56942 4.31943 8.38887 3.83332 6.99998 3.83332H6.70508L7.8301 4.95834L6.95192 5.83653L4.32373 3.20834L6.95192 0.580154L7.8301 1.45834L6.70508 2.58334H6.99998C8.74356 2.58334 10.2211 3.18911 11.4326 4.40065C12.6442 5.61218 13.25 7.08974 13.25 8.83332C13.25 10.4017 12.7366 11.7703 11.7099 12.9391C10.6832 14.1079 9.39099 14.8087 7.83331 15.0416Z"
                        />
                    </svg>
                </span>
            </x-button>

            {{-- <x-dropdown.dropdown
                anchor="end"
                offsetY="13px"
             >
                <x-slot:trigger
                    class="text-2xs"
                >
                    @lang('Share')
                    <span class="size-[34px] inline-grid place-content-center rounded-full border">
                        <x-tabler-share class="size-4" />
                    </span>
                </x-slot:trigger>

                <x-slot:dropdown
                    class="min-w-52 overflow-hidden p-2"
                 >
                    <x-button
                        class="w-full justify-start rounded-md px-3 py-2 text-start text-2xs hover:bg-heading-foreground/5"
                        variant="link"
                        target="_blank"
                        href="http://twitter.com/share?text={{ $workbook->output }}"
                    >
                        <x-tabler-brand-x class="size-6" />
                        @lang('X')
                    </x-button>
                    <x-button
                        class="w-full justify-start rounded-md px-3 py-2 text-start text-2xs hover:bg-heading-foreground/5"
                        variant="link"
                        target="_blank"
                        href="https://wa.me/?text={{ htmlspecialchars($workbook->output) }}"
                    >
                        <x-tabler-brand-whatsapp class="size-6" />
                        @lang('Whatsapp')
                    </x-button>
                    <x-button
                        class="w-full justify-start rounded-md px-3 py-2 text-start text-2xs hover:bg-heading-foreground/5"
                        variant="link"
                        target="_blank"
                        href="https://t.me/share/url?url={{ request()->host() }}&text={{ htmlspecialchars($workbook->output) }}"
                    >
                        <x-tabler-brand-telegram class="size-6" />
                        @lang('Telegram')
                    </x-button>
                </x-slot:dropdown>
            </x-dropdown.dropdown> --}}

            <x-button
                class="text-2xs"
                variant="link"
                @click.prevent="downloadImage(editingImage.output, editingImage.input)"
            >
                @lang('Download')
                <span class="size-[34px] inline-grid place-content-center rounded-full border">
                    <x-tabler-circle-chevron-down class="size-5" />
                </span>
            </x-button>

            <x-button
                class="text-2xs"
                variant="link"
                @click.prevent="showImageDetails = !showImageDetails; zoomLevel = 1;"
            >
                @lang('Details')
                <x-tabler-dots class="size-5 opacity-70" />
            </x-button>
        </div>

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
        class="size-[34px] hidden shrink-0"
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
