<div class="mb-8 border-b pb-6">
    <h3 class="mb-5 flex flex-wrap items-center justify-between gap-3">
        @lang('Review Your Content')
    </h3>
    <p>
        @lang('Start by selecting a company and product or create new ones at BrandCenter in a few clicks.')
    </p>
</div>

<form
    class="flex flex-col gap-6"
    id="stepsForm"
    action="{{ LaravelLocalization::localizeUrl(route('dashboard.user.automation.postindex')) }}"
    method="POST"
>
    @csrf
    <input
        type="hidden"
        name="platform_id"
        value="{{ $platform_id }}"
    />
    <input
        type="hidden"
        name="company_id"
        value="{{ $company_id }}"
    />
    @foreach ($product_id as $pid)
        <input
            type="hidden"
            name="product_id[]"
            value="{{ $pid }}"
        >
    @endforeach
    <input
        type="hidden"
        name="camp_id"
        value="{{ $camp_id }}"
    />
    <input
        type="hidden"
        name="camp_target"
        value="{{ $camp_target }}"
    />
    @foreach ($topics as $topic)
        <input
            type="hidden"
            name="topics[]"
            value="{{ $topic }}"
        >
    @endforeach
    <input
        type="hidden"
        name="seo"
        value="{{ $seo }}"
    />
    <input
        type="hidden"
        name="is_img"
        value="{{ $is_img }}"
    />
    <input
        type="hidden"
        name="tone"
        value="{{ $tone }}"
    />
    <input
        type="hidden"
        name="num_res"
        value="{{ $num_res }}"
    />
    <input
        type="hidden"
        name="vis_format"
        value="{{ $vis_format }}"
    />
    <input
        type="hidden"
        name="vis_ratio"
        value="{{ $vis_ratio }}"
    />
    <input
        id="cam_injected_name"
        type="hidden"
        name="cam_injected_name"
        value="{{ $cam_injected_name }}"
    />

    <input
        type="hidden"
        name="step"
        value="6"
    />

    <div
        class="grow md:flex-1"
        id="thePost"
    >
    </div>

    <x-forms.input
        type="checkbox"
        label="{{ __('Send a copy to my email address') }}"
        name="sendMail"
        type="checkbox"
        switcher
    />

    <x-button
        id="reviweNextBtn"
        variant="secondary"
        type="submit"
    >
        {{ __('Next') }}
        <span class="size-7 inline-grid place-items-center rounded-full bg-background text-foreground">
            <x-tabler-chevron-right class="size-4" />
        </span>
    </x-button>
</form>
