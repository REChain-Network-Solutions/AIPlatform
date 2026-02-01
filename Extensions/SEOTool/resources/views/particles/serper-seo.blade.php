<x-card
    class="mb-5 max-md:text-center"
    szie="lg"
>
    <label class="form-label mb-4">{{ __('Activate Serper SEO On:') }}</label>

    <x-card
        class="mb-2 max-md:text-center"
        szie="lg"
    >
        <label class="form-label mb-4">{{ __('For User:') }}</label>
        <div class="mb-1 grid grid-cols-2 gap-4 md:grid-cols-3">
            <x-forms.input
                class:container="h-full bg-input-background"
                class:label="w-full border h-full rounded px-3 py-4 hover:bg-foreground/5 transition-colors"
                class="checked-item"
                id="serper_seo_aw_keyword"
                :checked="setting('serper_seo_aw_keyword', 0) == 1"
                type="checkbox"
                name="serper_seo_aw_keyword"
                label="{{ __('Article Wizard Keywords') }}"
                custom
            />

            <x-forms.input
                class:container="h-full bg-input-background"
                class:label="w-full border h-full rounded px-3 py-4 hover:bg-foreground/5 transition-colors"
                class="checked-item"
                id="serper_seo_aw_sq"
                :checked="setting('serper_seo_aw_sq', 0) == 1"
                type="checkbox"
                name="serper_seo_aw_sq"
                label="{{ __('Article Wizard Search Questions') }}"
                custom
            />

            {{-- <x-forms.input
                class:container="h-full bg-input-background"
                class:label="w-full border h-full rounded px-3 py-4 hover:bg-foreground/5 transition-colors"
                class="checked-item"
                id="serper_seo_aw_anlyze"
                :checked="setting('serper_seo_aw_anlyze', 0) == 1 ? true : false"
                type="checkbox"
                name="serper_seo_aw_anlyze"
                label="{{ __('Article Wizard Analyze SEO Chart') }}"
                custom
            /> --}}

            {{-- <x-forms.input
                class:container="h-full bg-input-background"
                class:label="w-full border h-full rounded px-3 py-4 hover:bg-foreground/5 transition-colors"
                class="checked-item"
                id="serper_seo_aw_improve"
                :checked="setting('serper_seo_aw_improve', 0) == 1 ? true : false"
                type="checkbox"
                name="serper_seo_aw_improve"
                label="{{ __('Article Wizard Improve SEO Button') }}"
                custom
            /> --}}

            <x-forms.input
                class:container="h-full bg-input-background"
                class:label="w-full border h-full rounded px-3 py-4 hover:bg-foreground/5 transition-colors"
                class="checked-item"
                id="seo_ai_tool"
                :checked="setting('seo_ai_tool', 1) == 1"
                type="checkbox"
                name="seo_ai_tool"
                label="{{ __('SEO Tool') }}"
                custom
            />
            <x-forms.input
                class:container="h-full bg-input-background"
                class:label="w-full border h-full rounded px-3 py-4 hover:bg-foreground/5 transition-colors"
                class="checked-item"
                id="serper_seo_tool_improve"
                :checked="setting('serper_seo_tool_improve', 0) == 1"
                type="checkbox"
                name="serper_seo_tool_improve"
                label="{{ __('SEO Tool Improve SEO Button') }}"
                custom
            />
        </div>
    </x-card>
    <x-card
        class="mb-2 max-md:text-center"
        szie="lg"
    >
        <label class="form-label mb-4">{{ __('For Admin:') }}</label>
        <div class="mb-1 grid grid-cols-2 gap-4 md:grid-cols-3">
            <x-forms.input
                class:container="h-full bg-input-background"
                class:label="w-full border h-full rounded px-3 py-4 hover:bg-foreground/5 transition-colors"
                class="checked-item"
                id="serper_seo_blog_title_desc"
                :checked="setting('serper_seo_blog_title_desc', 0) == 1"
                type="checkbox"
                name="serper_seo_blog_title_desc"
                label="{{ __('Blog Title and Description') }}"
                custom
            />

            <x-forms.input
                class:container="h-full bg-input-background"
                class:label="w-full border h-full rounded px-3 py-4 hover:bg-foreground/5 transition-colors"
                class="checked-item"
                id="serper_seo_site_meta"
                :checked="setting('serper_seo_site_meta', 0) == 1"
                type="checkbox"
                name="serper_seo_site_meta"
                label="{{ __('Site Meta') }}"
                custom
            />
        </div>
    </x-card>

</x-card>
