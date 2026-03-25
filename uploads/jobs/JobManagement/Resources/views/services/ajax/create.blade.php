<link rel="stylesheet" href="{{ asset('vendor/css/dropzone.min.css') }}">

<div class="row">
    <div class="col-sm-12">
        <x-form id="save-item-form">
            @include('sections.password-autocomplete-hide')
            <div class="add-client bg-white rounded">
                <h4 class="mb-0 p-20 f-21 font-weight-normal text-capitalize border-bottom-grey">
                    @lang('modules.invoices.addService')</h4>
                <div class="row p-20">
                    <div class="col-lg-12">
                        <div class="row">
                            <input type="hidden" id="hiddenItemId">
                            <div class="col-lg-4 col-md-6">
                                <x-forms.text fieldId="name" :fieldLabel="__('app.name')" fieldName="name" fieldRequired="true"
                                    :fieldPlaceholder="__('--')">
                                </x-forms.text>
                            </div>

                            <div class="col-lg-4 col-md-6">
                                <x-forms.number class="mr-0 mr-lg-2 mr-md-2" :fieldLabel="__('app.price')" fieldName="price"
                                    fieldId="price" fieldRequired="true" :fieldPlaceholder="__('placeholders.price')" />
                            </div>

                            <div class="col-lg-4 col-md-6">
                                <x-forms.label class="my-3" fieldId="" :fieldLabel="__('engineerings::modules.itemCategory.itemCategory')">
                                </x-forms.label>
                                <x-forms.input-group>
                                    <select class="form-control select-picker" name="category_id" id="item_category_id"
                                        data-live-search="true">
                                        <option value="">--</option>
                                        @foreach ($categories as $category)
                                            <option value="{{ $category->id }}">
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


                            <div class="col-lg-4 col-md-6">
                                <x-forms.label class="my-3" fieldId="" :fieldLabel="__('engineerings::modules.itemCategory.itemSubCategory')">
                                </x-forms.label>
                                <x-forms.input-group>
                                    <select class="form-control select-picker" name="sub_category_id"
                                        id="sub_category_id" data-live-search="true">
                                        <option value="">@lang('engineerings::messages.noItemSubCategoryAdded')</option>
                                        @foreach ($subCategories as $subCategory)
                                            <option value="{{ $subCategory->id }}">
                                                {{ mb_ucwords($subCategory->name) }}</option>
                                        @endforeach
                                    </select>

                                    <x-slot name="append">
                                        <button id="add-sub-category" type="button" data-toggle="tooltip"
                                            data-original-title="{{ __('app.add') . ' ' . __('engineerings::modules.itemCategory.itemSubCategory') }}"
                                            class="btn btn-outline-secondary border-grey">@lang('app.add')</button>
                                    </x-slot>
                                </x-forms.input-group>
                            </div>

                            <div class="col-lg-4 col-md-6">
                                <x-forms.label class="my-3" fieldId="" :fieldLabel="__('modules.invoices.tax')">
                                </x-forms.label>
                                <x-forms.input-group>
                                    <select class="form-control select-picker" name="tax[]" id="tax_id"
                                        data-live-search="true" multiple="true">
                                        @foreach ($taxes as $tax)
                                            <option value="{{ $tax->id }}">{{ strtoupper($tax->tax_name) }}:
                                                {{ $tax->rate_percent }}%
                                            </option>
                                        @endforeach
                                    </select>

                                    @if (user()->permission('manage_tax') == 'all')
                                        <x-slot name="append">
                                            <button id="add-tax" type="button" data-toggle="tooltip"
                                                data-original-title="{{ __('app.add') . ' ' . __('modules.invoices.tax') }}"
                                                class="btn btn-outline-secondary border-grey">@lang('app.add')</button>
                                        </x-slot>
                                    @endif
                                </x-forms.input-group>
                            </div>

                            <div class="col-lg-4 col-md-6">
                                <x-forms.label class="my-3" fieldId="" :fieldLabel="__('modules.unitType.unitType')">
                                </x-forms.label>
                                <x-forms.input-group>
                                    <select class="form-control select-picker" name="unit_type" id="unit_type_id"
                                        data-live-search="true">
                                        @foreach ($unit_types as $unit_type)
                                            <option @if ($unit_type->default == 1) selected @endif
                                                value="{{ $unit_type->id }}">{{ ucwords($unit_type->unit_type) }}
                                            </option>
                                        @endforeach
                                    </select>
                                </x-forms.input-group>
                            </div>

                            <div class="col-lg-3 col-md-6 mt-3">
                                <x-forms.checkbox class="mr-0 mr-lg-2 mr-md-2" :fieldLabel="__('app.purchaseAllow')"
                                    fieldName="purchase_allow" fieldId="purchase_allow" fieldValue="no"
                                    fieldRequired="true" />
                            </div>
                            <div class="col-md-12 mt-3">
                                <div class="form-group">
                                    <x-forms.label class="my-3" fieldId="description-text" :fieldLabel="__('app.description')">
                                    </x-forms.label>
                                    <textarea name="description" id="description-text" rows="4" class="form-control"></textarea>
                                </div>
                            </div>
                            <div class="col-lg-12">
                                <x-forms.file allowedFileExtensions="png jpg jpeg svg"
                                    class="mr-0 mr-lg-2 mr-md-2 cropper" :fieldLabel="__('engineerings::app.menu.image')" fieldName="image"
                                    fieldId="image" />
                            </div>
                            <input type ="hidden" name="add_more" value="false" id="add_more" />
                        </div>
                    </div>
                </div>

                <x-form-actions>
                    <x-forms.button-primary id="save-item" class="mr-3" icon="check">@lang('app.save')
                    </x-forms.button-primary>
                    <x-forms.button-cancel :link="route('engineerings.create')" class="border-0">@lang('app.cancel')
                    </x-forms.button-cancel>
                </x-form-actions>
            </div>
        </x-form>

    </div>
</div>

<script src="{{ asset('vendor/jquery/dropzone.min.js') }}"></script>
<script>
    $(document).ready(function() {
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

        $('#save-item').click(function() {
            const url = "{{ route('services.store') }}";
            var data = $('#save-item-form').serialize();
            saveItem(data, url, "#save-item");
        });

        function saveItem(data, url, buttonSelector) {
            $.easyAjax({
                url: url,
                container: '#save-item-form',
                type: "POST",
                disableButton: true,
                blockUI: true,
                buttonSelector: buttonSelector,
                file: true,
                data: data,
                success: function(response) {
                    if (response.add_more == true) {
                        var right_modal_content = $.trim($(RIGHT_MODAL_CONTENT).html());
                        var responseOptions = response.data;
                        $('select[name="services_id[]"]').append(responseOptions);
                        $('select[name="services_id[]"]').selectpicker('refresh');

                        if (right_modal_content.length) {
                            $(RIGHT_MODAL_CONTENT).html(response.html.html);
                            $('#add_more').val(false);
                        } else {
                            $('.content-wrapper').html(response.html.html);
                            init('.content-wrapper');
                            $('#add_more').val(false);
                        }
                    } else {
                        if ($(MODAL_XL).hasClass('show')) {
                            $(MODAL_XL).modal('hide');
                            window.location.reload();
                        } else {
                            window.location.href = response.redirectUrl;
                        }
                    }

                    if (typeof showTable !== 'undefined' && typeof showTable === 'function') {
                        showTable();
                    }
                }
            });
        }

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
