<div class="lqd-adv-editor-enhance py-9">
    <div class="mb-6 flex items-center justify-between gap-3">
        <h2 class="mb-0">
            @lang('Enhance Your Images')
        </h2>

        <x-button
            class="text-2xs font-medium opacity-80 hover:opacity-100"
            variant="link"
            href="#"
        >
            @lang('View All')
            <x-tabler-chevron-right class="size-4" />
        </x-button>
    </div>

    <div class="lqd-adv-editor-enhance-grid grid grid-cols-1 gap-5 sm:grid-cols-2 md:grid-cols-3 md:gap-x-6 lg:grid-cols-4 lg:gap-x-9 lg:gap-y-12">
        @foreach ($tools as $tool)
            <div
                class="lqd-adv-editor-enhance-grid-item group/item relative overflow-hidden rounded-lg shadow-md shadow-black/5 transition-all hover:shadow-[0_38px_55px_rgba(0,0,0,0.1)]">
                <figure class="relative mb-0 w-full bg-heading-foreground/5 transition-all">
                    <img
                        class="transition-all group-hover/item:scale-105"
                        src="{{ custom_theme_url($tool['image']) }}"
                        alt="{{ $tool['title'] }}"
                    />
                    @if ($tool['premium'])
                        <span class="absolute start-5 top-5 z-2 inline-flex items-center gap-1.5 rounded bg-secondary px-1.5 py-2 text-3xs font-medium text-secondary-foreground">
                            <svg
                                width="19"
                                height="15"
                                viewBox="0 0 19 15"
                                fill="none"
                                stroke="currentColor"
                                xmlns="http://www.w3.org/2000/svg"
                            >
                                <path
                                    d="M7.74854 7.49995L5.99854 5.57495L6.52354 4.69995M4.24854 1.375H14.7485L17.3735 5.75L9.93604 14.0625C9.87901 14.1207 9.81094 14.1669 9.73581 14.1985C9.66069 14.2301 9.58002 14.2463 9.49854 14.2463C9.41705 14.2463 9.33638 14.2301 9.26126 14.1985C9.18613 14.1669 9.11806 14.1207 9.06104 14.0625L1.62354 5.75L4.24854 1.375Z"
                                    stroke-width="1.5"
                                    stroke-linecap="round"
                                    stroke-linejoin="round"
                                />
                            </svg>
                            @lang('Premium')
                        </span>
                    @endif
                </figure>
                <div class="px-6 pb-6 pt-8">
                    <h5 class="mb-2 text-base font-semibold">
                        {{ $tool['title'] }}
                    </h5>
                    <p class="text-2xs/5 opacity-60">
                        {{ $tool['description'] }}
                    </p>
                </div>

                <a
                    class="absolute inset-0"
                    href="#"
                    @click.prevent="editingImage = {}; selectedTool = '{{ $tool['action'] }}'; switchToolsCat({toolKey: '{{ $tool['action'] }}'}); switchView('editor');"
                >
                    <span class="sr-only">View Template</span>
                </a>
            </div>
        @endforeach
    </div>
</div>
