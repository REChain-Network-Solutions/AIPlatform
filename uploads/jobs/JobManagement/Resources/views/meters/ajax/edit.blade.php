@extends('layouts.app')

@section('content')
    <div class="content-wrapper">
        <div class="bg-white rounded b-shadow-4 create-inv">
            <!-- HEADING START -->
            <div class="px-lg-4 px-md-4 px-3 py-3">
                <h4 class="mb-0 f-21 font-weight-normal text-capitalize">@lang('engineerings::modules.meter') @lang('app.details')</h4>
            </div>
            <!-- HEADING END -->
            <hr class="m-0 border-top-grey">
            <!-- FORM START -->
            <x-form class="c-inv-form" id="saveInvoiceForm"> @method('PUT')
                <div class="row px-lg-4 px-md-4 px-3 py-3">
                    <div class="col-md-3">
                        <div class="form-group mb-lg-0 mb-md-0 mb-4">
                            <x-forms.label fieldId="billing_date" :fieldLabel="__('engineerings::app.menu.billingDate')" fieldRequired="true">
                            </x-forms.label>
                            <div class="input-group">
                                <input type="text" id="billing_date" name="billing_date"
                                    class="px-6 position-relative text-dark font-weight-normal form-control-plaintext height-35 rounded p-0 text-left f-15"
                                    placeholder="@lang('placeholders.date')"
                                    value="{{ \Carbon\Carbon::createFromFormat('Y-m-d', $meter->billing_date)->format('d-m-Y') }}"
                                    readonly>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="form-group mb-lg-0 mb-md-0 mb-4">
                            <x-forms.label fieldId="type_bill" :fieldLabel="__('engineerings::app.menu.typeBill')" fieldName="type_bill"
                        fieldRequired="true">
                    </x-forms.label>
                            <x-forms.input-group>
                                <select class="form-control select-picker" name="type_bill" id="type_bill"
                                    data-live-search="true">
                                    <option @if ('el' == $meter->type_bill) selected @endif value="el">EL</option>
                                    <option @if ('wt' == $meter->type_bill) selected @endif value="wt">WT</option>
                                    <option @if ('gaz' == $meter->type_bill) selected @endif value="gaz">GAZ</option>
                                </select>
                            </x-forms.input-group>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="form-group mb-lg-0 mb-md-0 mb-4">
                            <x-forms.label fieldId="unit_id" :fieldLabel="__('engineerings::app.menu.unitID')" fieldName="unit_id" fieldRequired="true">
                            </x-forms.label>
                            <x-forms.input-group>
                                <select class="form-control select-picker" name="unit_id" id="unit_id"
                                    data-live-search="true">
                                    @foreach ($unit as $items)
                                        <option @if ($items->unit_code == $meter->unit_id) selected @endif
                                            value="{{ $items->unit_code }}">{{ ucwords($items->unit_name) }}</option>
                                    @endforeach
                                </select>
                            </x-forms.input-group>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="form-group mb-lg-0 mb-md-0 mb-4">
                            <x-forms.label fieldId="end_meter" :fieldLabel="__('engineerings::app.menu.endMeter')" fieldName="end_meter" fieldRequired="true">
                            </x-forms.label>
                            <x-forms.input-group>
                                <input type="number" id="end_meter" name="end_meter"
                                    class="px-6 position-relative text-dark font-weight-normal form-control height-35 rounded p-0 text-left f-15"
                                    value="{{ $meter->end_meter }}">
                            </x-forms.input-group>
                        </div>
                    </div>

                    <div class="col-lg-12">
                        <x-forms.file allowedFileExtensions="png jpg jpeg svg" class="mr-0 mr-lg-2 mr-md-2 cropper"
                            :fieldLabel="__('engineerings::app.menu.image')" :fieldValue="$meter->image_url" fieldName="image" fieldId="image"/>
                    </div>
                </div>
                <!-- CANCEL SAVE START -->
                <x-form-actions class="c-inv-btns d-block d-lg-flex d-md-flex">
                    <x-forms.button-primary id="save-unit-form" class="save-form mr-3" icon="check">@lang('app.save') Meter
                    </x-forms.button-primary>

                    <x-forms.button-cancel :link="route('meter.index')" class="border-0 ">@lang('app.cancel')
                    </x-forms.button-cancel>
                </x-form-actions>
                <!-- CANCEL SAVE END -->

            </x-form>
            <!-- FORM END -->
        </div>
    </div>
@endsection

@push('scripts')
    <script>
        $(document).ready(function() {
            if ($('.custom-date-picker').length > 0) {
                datepicker('.custom-date-picker', {
                    position: 'bl',
                    ...datepickerConfig
                });
            }

            const dp1 = datepicker('#billing_date', {
                position: 'bl',
                onSelect: (instance, date) => {
                    var rotation = $('#rotation').val();
                    nextDate(rotation, date);
                },
                dateSelected: new Date("{{ str_replace('-', '/', $meter->billing_date) }}"),
                ...datepickerConfig
            });

            $('.save-form').click(function() {
                const url = "{{ route('meter.update', $meter->id) }}";
                $.easyAjax({
                    url: url,
                    container: '#saveInvoiceForm',
                    type: "POST",
                    disableButton: true,
                    blockUI: true,
                    buttonSelector: "#save-unit-form",
                    file: true,
                    data: $('#saveInvoiceForm').serialize(),
                    success: function (response) {
                    if (response.status === 'success') {
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
            
            init(RIGHT_MODAL);
        });
    </script>
@endpush
