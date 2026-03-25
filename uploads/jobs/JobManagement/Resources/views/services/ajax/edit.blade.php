<link rel="stylesheet" href="{{ asset('vendor/css/dropzone.min.css') }}">

<div class="row">
    <div class="col-sm-12">
        <x-form id="save-item-data-form" method="PUT">
            <div class="add-client bg-white rounded">
                <h4 class="mb-0 p-20 f-21 font-weight-normal text-capitalize border-bottom-grey">
                    @lang('modules.invoices.editService')</h4>
                <div class="row p-20">
                    <div class="col-lg-12">
                        <div class="row">
                            <div class="col-md-4">
                                <x-forms.text fieldId="name" :fieldLabel="__('app.name')" fieldName="name" fieldRequired="true"
                                    :fieldPlaceholder="__('placeholders.itemName')" :fieldValue="$services->name">
                                </x-forms.text>
                            </div>

                            <div class="col-md-4">
                                <x-forms.number class="mr-0 mr-lg-2 mr-md-2" :fieldLabel="__('app.price')" fieldName="price"
                                    fieldId="price" :fieldPlaceholder="__('placeholders.price')" :fieldValue="$services->price" />
                            </div>

                            <div class="col-lg-4 col-md-6">
                                <x-forms.label class="my-3" fieldId="" :fieldLabel="__('engineerings::modules.itemCategory.itemCategory')">
                                </x-forms.label>
                                <x-forms.input-group>
                                    <select class="form-control select-picker" name="category_id" id="item_category_id"
                                        data-live-search="true">
                                        <option value="">--</option>
                                        @foreach ($categories as $category)
                                            <option value="{{ $category->id }}"
                                                @if ($category->id == $services->category_id) selected @endif>
                                                {{ mb_ucwords($category->name) }}</option>
                                        @endforeach
                                    </select>

                                    <x-slot name="append">
                                        <button id="add-category" type="button" data-toggle="tooltip"
                                            data-original-title="{{ __('app.add') . ' ' . __('engineerings::modules.itemCategory.itemCategory') }}"
                                            class="btn btn-outline-secondary border-grey">@lang('app.add')</button>
                                    </x-slot>
                                </x-forms.input-group>
                            </div>

                            <div class="col-md-4">
                                <x-forms.label class="my-3" fieldId="" :fieldLabel="__('engineerings::modules.itemCategory.itemSubCategory')">
                                </x-forms.label>
                                <x-forms.input-group>
                                    <select class="form-control select-picker" name="sub_category_id"
                                        id="sub_category_id" data-live-search="true">
                                        <option value="">@lang('engineerings::messages.noItemSubCategoryAdded')</option>
                                        @if ($services->category_id)
                                            @foreach ($subCategories as $subCategory)
                                                <option value="{{ $subCategory->id }}"
                                                    @if ($category->id == $services->sub_category_id) selected @endif>
                                                    {{ mb_ucwords($subCategory->name) }}</option>
                                            @endforeach
                                        @endif
                                    </select>

                                    <x-slot name="append">
                                        <button id="add-sub-category" type="button"
                                            class="btn btn-outline-secondary border-grey" data-toggle="tooltip"
                                            data-original-title="{{ __('app.add') . ' ' . __('modules.itemCategory.itemSubCategory') }}">@lang('app.add')</button>
                                    </x-slot>
                                </x-forms.input-group>
                            </div>

                            <div class="col-md-4">
                                <x-forms.label class="my-3" fieldId="multiselect" :fieldLabel="__('modules.invoices.tax')">
                                </x-forms.label>
                                <x-forms.input-group>
                                    <select class="form-control select-picker" name="tax[]" id="multiselect"
                                        data-live-search="true" multiple="true">
                                        @foreach ($taxes as $tax)
                                            <option value="{{ $tax->id }}"
                                                @if (isset($services->taxes) && array_search($tax->id, json_decode($services->taxes)) !== false) selected @endif>
                                                {{ strtoupper($tax->tax_name) }}: {{ $tax->rate_percent }}%
                                            </option>
                                        @endforeach
                                    </select>

                                    @if (user()->permission('manage_tax') == 'all')
                                        <x-slot name="append">
                                            <button id="add-tax" type="button"
                                                class="btn btn-outline-secondary border-grey" data-toggle="tooltip"
                                                data-original-title="{{ __('app.add') . ' ' . __('modules.invoices.tax') }}">@lang('app.add')</button>
                                        </x-slot>
                                    @endif
                                </x-forms.input-group>
                            </div>

                            <div class="col-lg-3 col-md-6">
                                <x-forms.label class="my-3" fieldId="" :fieldLabel="__('modules.unitType.unitType')">
                                </x-forms.label>
                                <x-forms.input-group>
                                    <select class="form-control select-picker" name="unit_type" id="unit_type_id"
                                        data-live-search="true" multiple="true">
                                        @foreach ($unit_types as $unit_type)
                                            <option value="{{ $unit_type->id }}"
                                                @if ($unit_type->id == $services->unit_id) selected @endif>
                                                {{ $unit_type->unit_type }}
                                            </option>
                                        @endforeach
                                    </select>
                                </x-forms.input-group>
                            </div>

                            <div class="col-md-3 col-md-6 mt-3">
                                <x-forms.checkbox class="mr-0 mr-lg-2 mr-md-2" :fieldLabel="__('app.purchaseAllow')"
                                    fieldName="purchase_allow" fieldId="purchase_allow" fieldValue="no"
                                    fieldRequired="true" :checked="$services->allow_purchase == 1" />
                            </div>

                            <div class="col-md-12 mt-3">
                                <div class="form-group">
                                    <x-forms.label class="my-3" fieldId="description-text" :fieldLabel="__('app.description')">
                                    </x-forms.label>
                                    <textarea name="description" id="description-text" rows="4" class="form-control f-14 w-100">{{ $services->description }}</textarea>
                                </div>
                            </div>

                            <div class="col-lg-12">
                                <x-forms.file allowedFileExtensions="png jpg jpeg svg"
                                    class="mr-0 mr-lg-2 mr-md-2 cropper" :fieldLabel="__('engineerings::app.menu.image')" :fieldValue="$services->image_url"
                                    fieldName="image" fieldId="image" />
                            </div>

                        </div>
                    </div>
                </div>

                <x-form-actions>
                    <x-forms.button-primary id="save-item-form" class="mr-3" icon="check">@lang('app.save')
                    </x-forms.button-primary>
                    <x-forms.button-cancel :link="route('engineerings.create')" class="border-0">@lang('app.cancel')
                    </x-forms.button-cancel>
                </x-form-actions>
            </div>
        </x-form>
    </div>
</div>

<script>
    $(document).ready(function() {
        $('#save-item-form').click(function() {
            const url = "{{ route('services.update', [$services->id]) }}";
            $.easyAjax({
                url: url,
                container: '#save-item-data-form',
                type: "POST",
                disableButton: true,
                blockUI: true,
                buttonSelector: "#save-item-form",
                file: true,
                data: $('#save-item-data-form').serialize(),
                success: function(response) {
                    if (response.add_more == true) {
                        var right_modal_content = $.trim($(RIGHT_MODAL_CONTENT).html());
                        var responseOptions = response.data;
                        $('select[name="services_id[]"]').append(responseOptions);
                        $('select[name="services_id[]"]').selectpicker('refresh');

                        if (right_modal_content.length) {
                            $(RIGHT_MODAL_CONTENT).html(response.html);
                        } else {
                            $('.content-wrapper').html(response.html);
                            init('.content-wrapper');
                        }
                    } else {
                        if ($(MODAL_XL).hasClass('show')) {
                            $(MODAL_XL).modal('hide');
                            window.location.reload();
                        } else {
                            window.location.href = response.redirectUrl;
                        }
                    }
                }
            });
        });

        $('#item_category_id').change(function(e) {
            let categoryId = $(this).val();
            let url = "{{ route('get_services_sub_category', ':id') }}";
            url = (categoryId) ? url.replace(':id', categoryId) : url.replace(':id', null);
            $.easyAjax({
                url: url,
                type: "GET",
                success: function(response) {
                    if (response.status == 'success') {
                        var options = [];
                        var rData;
                        rData = response.data;
                        $.each(rData, function(index, value) {
                            var selectData;
                            selectData = '<option value="' + value.id + '">' + value
                                .name + '</option>';
                            options.push(selectData);
                        });

                        $('#sub_category_id').html('<option value="">--</option>' +
                            options);
                        $('#sub_category_id').selectpicker('refresh');
                    }
                }
            })
        });

        $('#add-category').click(function() {
            const url = "{{ route('services-category.create') }}";
            $(MODAL_LG + ' ' + MODAL_HEADING).html('...');
            $.ajaxModal(MODAL_LG, url);
        })

        $('#add-sub-category').click(function() {
            const url = "{{ route('sub-services-category.create') }}";
            $(MODAL_LG + ' ' + MODAL_HEADING).html('...');
            $.ajaxModal(MODAL_LG, url);
        });

        $('#add-tax').click(function() {
            const url = "{{ route('taxes.create') }}";
            $(MODAL_LG + ' ' + MODAL_HEADING).html('...');
            $.ajaxModal(MODAL_LG, url);
        });

        init(RIGHT_MODAL);
    });
</script>
