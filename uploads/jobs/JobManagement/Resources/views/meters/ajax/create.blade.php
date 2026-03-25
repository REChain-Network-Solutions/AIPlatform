<style>
    #reader {
        width: 500px;
    }

    .result {
        background-color: green;
        color: #fff;
        padding: 20px;
    }

    #reader__scan_region {
        background: white;
    }
</style>

<link rel="stylesheet" href="{{ asset('vendor/css/dropzone.min.css') }}">
<!-- CREATE INVOICE START -->
<div class="bg-white rounded b-shadow-4 create-inv">
    <!-- HEADING START -->
    <div class="px-lg-4 px-md-4 px-3 py-3">
        <h4 class="mb-0 f-21 font-weight-normal text-capitalize">@lang('engineerings::modules.meter') @lang('app.details')</h4>
    </div>
    <!-- HEADING END -->
    <hr class="m-0 border-top-grey">
    <!-- FORM START -->
    <x-form class="c-inv-form" id="saveInvoiceForm">
        <div class="row px-lg-4 px-md-4 px-3 py-3">
            <div class="col-md-3">
                <div class="form-group mb-lg-0 mb-md-0 mb-4">
                    <x-forms.label fieldId="billing_date" :fieldLabel="__('engineerings::app.menu.billingDate')" fieldRequired="true">
                    </x-forms.label>
                    <div class="input-group">
                        <input type="text" id="billing_date" name="billing_date"
                            class="px-6 position-relative text-dark font-weight-normal form-control height-35 rounded p-0 text-left f-15"
                            placeholder="@lang('placeholders.date')"
                            value="{{ now(company()->timezone)->translatedFormat(company()->date_format) }}">
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="form-group mb-lg-0 mb-md-0 mb-4">
                    <x-forms.label fieldId="type_bill" :fieldLabel="__('engineerings::app.menu.typeBill')" fieldName="type_bill" fieldRequired="true">
                    </x-forms.label>
                    <x-forms.input-group>
                        <select class="form-control select-picker" name="type_bill" id="type_bill"
                            data-live-search="true">
                            <option value="el">EL</option>
                            <option value="wt">WT</option>
                            <option value="gaz">GAZ</option>
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
                            <option value="">--</option>
                            @foreach ($unit as $items)
                                <option value="{{ $items->id }}">{{ ucwords($items->unit_code) }}</option>
                            @endforeach
                        </select>
                        <x-slot name="append">
                            <button data-toggle="modal" data-target="#exampleModal" type="button"
                                class="btn btn-outline-secondary border-grey">
                                <svg xmlns="http://www.w3.org/2000/svg" width="26" height="26"
                                    fill="currentColor" class="bi bi-upc" viewBox="0 0 14 14">
                                    <path
                                        d="M3 4.5a.5.5 0 0 1 1 0v7a.5.5 0 0 1-1 0zm2 0a.5.5 0 0 1 1 0v7a.5.5 0 0 1-1 0zm2 0a.5.5 0 0 1 1 0v7a.5.5 0 0 1-1 0zm2 0a.5.5 0 0 1 .5-.5h1a.5.5 0 0 1 .5.5v7a.5.5 0 0 1-.5.5h-1a.5.5 0 0 1-.5-.5zm3 0a.5.5 0 0 1 1 0v7a.5.5 0 0 1-1 0z" />
                                </svg>
                            </button>
                        </x-slot>
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
                            placeholder="@lang('')">
                    </x-forms.input-group>
                </div>
            </div>

            <div class="col-lg-12">
                <x-forms.file allowedFileExtensions="png jpg jpeg svg" class="mr-0 mr-lg-2 mr-md-2 cropper"
                    :fieldLabel="__('engineerings::app.menu.image')" fieldName="image" fieldId="image" />
            </div>
        </div>

        <!-- CANCEL SAVE START -->
        <x-form-actions class="c-inv-btns d-block d-lg-flex d-md-flex">
            <x-forms.button-primary data-type="save" class="save-form mr-3" icon="check">@lang('app.save')
            </x-forms.button-primary>

            <x-forms.button-cancel :link="route('meter.index')" class="border-0 ">@lang('app.cancel')
            </x-forms.button-cancel>
        </x-form-actions>
        <!-- CANCEL SAVE END -->

    </x-form>
    <!-- FORM END -->
</div>

<script src="{{ asset('vendor/jquery/dropzone.min.js') }}"></script>
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
            ...datepickerConfig
        });

        $('.save-form').click(function() {
            $.easyAjax({
                url: "{{ route('meter.store') }}",
                container: '#saveInvoiceForm',
                type: "POST",
                disableButton: true,
                blockUI: true,
                file: true,
                data: $('#saveInvoiceForm').serialize(),
                success: function(response) {
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

<!-- MODAL -->
<div class="modal fade" id="exampleModal" tabindex="-1" role="dialog" aria-labelledby="exampleModalLabel"
    aria-hidden="true">
    <div class="modal-dialog" role="document">
        <div class="modal-content">
            <div class="modal-header">
                <h5 class="modal-title" id="exampleModalLabel">@lang('engineerings::modules.meter') @lang('app.details')</h5>
                <button type="button" class="close" data-dismiss="modal" aria-label="Close">
                    <span aria-hidden="true">&times;</span>
                </button>
            </div>
            <div class="modal-body">
                <div id="reader" style="width: 100%;"></div>
            </div>
            <div class="modal-footer">
                <button type="button" class="btn btn-secondary" data-dismiss="modal">Close</button>
            </div>
        </div>
    </div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/html5-qrcode/2.2.0/html5-qrcode.min.js"></script>
<script>
    let html5QrCodeScanner = new Html5QrcodeScanner("reader", {
        fps: 10,
        qrbox: 300
    });

    // html5QrCodeScanner.render(onScanSuccess, onScanError);

    $('#exampleModal').on('show.bs.modal', function() {
        html5QrCodeScanner.render(onScanSuccess, onScanError);

        document.getElementById('reader__dashboard_section_csr').style.display = 'block';
        document.getElementById('reader__dashboard_section_fsr').style.display = 'none';

        document.getElementById('reader__dashboard_section_swaplink').addEventListener('click', function(e) {
            e.preventDefault();
        });
    });

    $('#exampleModal').on('hide.bs.modal', function() {
        html5QrCodeScanner.clear().then(_ => {}).catch(error => {});
    });

    var unitOptions = [
        @foreach ($unit as $items)
            { value: '{{ $items->id }}', text: '{{ $items->unit_code }}' },
        @endforeach
    ];

    function onScanSuccess(qrCodeMessage) {
        var selectElement = document.getElementById('unit_id');
        var option = null;

        for (var i = 0; i < unitOptions.length; i++) {
            if (unitOptions[i].text.toLowerCase() === qrCodeMessage.toLowerCase()) {
                option = unitOptions[i];
                break;
            }
        }

        selectElement.value = option ? option.value : '';
        $(selectElement).selectpicker('refresh');

        if (!option) {
            Swal.fire({
                icon: 'error',
                title: 'Error',
                text: 'QR Code tidak sesuai dengan opsi yang ada.',
            });
        }

        html5QrCodeScanner.clear().then(_ => {}).catch(error => {});
        $('#exampleModal').modal('hide');
    }

    function onScanError(errorMessage) {}
</script>