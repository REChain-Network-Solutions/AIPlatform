<div class="mb-8 border-b pb-6">
    <h3 class="mb-5 flex flex-wrap items-center justify-between gap-3">
        @lang('Company Info')
        <x-button
            class="ms-auto"
            variant="secondary"
            href="{{ LaravelLocalization::localizeUrl(route('dashboard.user.brand.index')) }}"
        >
            {{ __('BrandCenter') }}
        </x-button>
    </h3>
    <p>
        @lang('Start by selecting a company or create a new one at BrandCenter in a few clicks.')
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
        name="step"
        value="3"
    />

    <x-forms.input
        label="{{ __('Select a Company') }}"
        size="lg"
        name="company_id"
        type="select"
    >
        <x-slot:label-extra>
            <x-button
                class="text-2xs font-semibold text-primary"
                variant="link"
                href="{{ LaravelLocalization::localizeUrl(route('dashboard.user.brand.index')) }}"
            >
                {{ __('Add New') }}
                <x-tabler-plus class="size-4" />
            </x-button>
        </x-slot:label-extra>

        <option value="0">
            {{ __('Select a Company') }}
        </option>
        @foreach (\App\Models\Company::getMyCompanies() ?? [] as $company)
            <option value="{{ $company->id }}">
                {{ $company->name }}
            </option>
        @endforeach
    </x-forms.input>

    <div class="space-y-2">
        <x-forms.input
            id="product_id[]"
            label="{{ __('Select a Product') }}"
            size="lg"
            name="product_id[]"
            type="select"
            size="lg"
            multiple
            required
        >
            <x-slot:label-extra>
                <x-modal title="{{ __('Choose a company') }}">
                    <x-slot:trigger
                        class="text-2xs font-semibold text-primary"
                        variant="link"
                    >
                        {{ __('Add New') }}
                        <x-tabler-plus class="size-4" />
                    </x-slot:trigger>
                    <x-slot:modal>
                        <div class="flex flex-col gap-2">
                            @foreach (\App\Models\Company::getMyCompanies() ?? [] as $company)
                                <a
                                    class="block rounded-md px-3 py-2 hover:bg-foreground/5"
                                    href="{{ LaravelLocalization::localizeUrl(route('dashboard.user.brand.edit', $company->id)) }}"
                                >
                                    {{ $company->name }}
                                </a>
                            @endforeach
                        </div>
                        <div class="mt-4 border-t pt-3">
                            <x-button
                                @click.prevent="modalOpen = false"
                                variant="outline"
                            >
                                {{ __('Cancel') }}
                            </x-button>
                        </div>
                    </x-slot:modal>
                </x-modal>
            </x-slot:label-extra>
        </x-forms.input>

        <x-alert>
            {{ __('You can select multiple products or services.') }}
        </x-alert>
    </div>

    <x-button
        size="lg"
        variant="secondary"
        type="submit"
    >
        {{ __('Next') }}
        <span class="size-7 inline-grid place-items-center rounded-full bg-background text-foreground">
            <x-tabler-chevron-right class="size-4" />
        </span>
    </x-button>
</form>

@push('script')
    <script src="{{ custom_theme_url('/assets/select2/select2.min.js') }}"></script>
    <script>
        $(document).ready(function() {
            "use strict";


            if ($.fn.select2) {
                $('.select2').select2({
                    tags: false
                });
            };


            var companySelect = $('select[name="company_id"]');
            var productSelect = $('select[name="product_id[]"]');

            // Attach an event handler to the company select element
            companySelect.on('change', function() {
                var selectedCompany = $(this).val();

                // Make an AJAX request to get products for the selected company
                $.get('/dashboard/user/automation/company/get-products/' + selectedCompany, function(data) {

                    productSelect.empty();
                    $.each(data, function(key, product) {
                        productSelect.append('<option value="' + product.id + '">' + product.name + '</option>');
                    });
                });
            });

            companySelect.trigger('change');
        });
    </script>
@endpush
