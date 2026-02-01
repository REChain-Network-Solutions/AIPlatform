<div class="lqd-adv-editor-recent-images py-9">
    <div class="mb-6 flex items-center justify-between gap-3">
        <h2 class="mb-0">
            @lang('Recent Images')
        </h2>

        <x-button
            class="text-2xs font-medium opacity-80 hover:opacity-100"
            variant="link"
            href="#"
            @click.prevent="switchView('gallery')"
        >
            @lang('View All')
            <x-tabler-chevron-right class="size-4" />
        </x-button>
    </div>


    <div
        class="lqd-adv-editor-recent-images-grid grid grid-cols-1 gap-5 sm:grid-cols-2 md:grid-cols-3 md:gap-x-6 lg:grid-cols-5 lg:gap-x-11 [&_.image-result:nth-child(n+7)]:hidden">
        @include('advanced-image::shared-components.image-grid', ['images' => $images, 'has_more' => false, 'id_prefix' => 'recent-'])
    </div>
</div>
