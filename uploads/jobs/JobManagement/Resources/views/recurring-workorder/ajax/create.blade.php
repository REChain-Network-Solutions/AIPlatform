<style>
    .customSequence .btn {
        border: none;
    }

    .information-box {
        border-style: dotted;
        margin-bottom: 30px;
        margin-top: 10px;
        padding-top: 10px;
        border-radius: 4px;
    }
</style>
<!-- CREATE INVOICE START -->
<div class="bg-white rounded b-shadow-4 create-inv">
    <!-- HEADING START -->
    <div class="px-lg-4 px-md-4 px-3 py-3">
        <h4 class="mb-0 f-21 font-weight-normal text-capitalize">@lang('engineerings::modules.recWorkOrders')</h4>
    </div>
    <!-- HEADING END -->
    <hr class="m-0 border-top-grey">
    <!-- FORM START -->
    <x-form class="c-inv-form" id="saveInvoiceForm">
        <div class="row px-lg-4 px-md-4 px-3 py-3">
            <div class="col-md-4">
                <div class="form-group mb-lg-0 mb-md-0 mb-4">
                    <x-forms.label fieldId="workrequest_id" :fieldLabel="__('engineerings::app.menu.WRid')" fieldName="workrequest_id"
                        fieldRequired="true">
                    </x-forms.label>
                    <x-forms.input-group>
                        <select class="form-control select-picker" name="workrequest_id" id="workrequest_id"
                            data-live-search="true">
                            @foreach ($wr as $items)
                                <option value="{{ $items->id }}">{{ ucwords($items->wr_no) }}</option>
                            @endforeach
                        </select>
                    </x-forms.input-group>
                </div>
            </div>
            <div class="col-lg-4">
                <x-forms.label fieldId="category" :fieldLabel="__('engineerings::app.menu.category')" fieldRequired="true"></x-forms.label>
                <x-forms.input-group>
                    <select class="form-control select-picker" name="category" id="category">
                        <option value="">--</option>
                        <option value="planned">Planned</option>
                        <option value="unplanned">Unplanned</option>
                    </select>
                </x-forms.input-group>
            </div>
            <div class="col-lg-4">
                <x-forms.label fieldId="priority" :fieldLabel="__('engineerings::app.menu.priority')" fieldRequired="true"></x-forms.label>
                <x-forms.input-group>
                    <select class="form-control select-picker" name="priority" id="priority">
                        <option value="">--</option>
                        <option value="low">Low</option>
                        <option value="medium">Medium</option>
                        <option value="high">High</option>
                        <option value="emergency">Emergency</option>
                    </select>
                </x-forms.input-group>
            </div>

            <div class="col-md-6">
                <x-forms.label class="mt-3" fieldId="parent_label" :fieldLabel="__('engineerings::app.menu.unit')" fieldName="parent_label">
                </x-forms.label>
                <select class="form-control select-picker" name="unit_id" id="unit_id" data-live-search="true">
                    <option value="">--</option>
                    @foreach ($unit as $items)
                        <option value="{{ $items->id }}">{{ $items->unit_name }}</option>
                    @endforeach
                </select>
            </div>
            <div class="col-lg-6">
                <x-forms.label class="mt-3" fieldId="assets_id" :fieldLabel="__('engineerings::app.menu.assets')"
                    fieldRequired="true"></x-forms.label>
                <x-forms.input-group>
                    <select class="form-control select-picker" name="assets_id" id="assets_id">
                        <option value="">--</option>
                    </select>
                </x-forms.input-group>
            </div>

            <div class="col-lg-4">
                <x-forms.label class="mt-3" fieldId="schedule_start" :fieldLabel="__('engineerings::app.menu.scheduleStart')" fieldRequired="true">
                </x-forms.label>
                <div class="bootstrap-timepicker timepicker">
                    <input type="datetime-local" id="schedule_start" name="schedule_start"
                        class="px-6 position-relative text-dark font-weight-normal form-control height-35 rounded p-0 text-left f-15">
                </div>
            </div>
            <div class="col-lg-4">
                <x-forms.label class="mt-3" fieldId="schedule_finish" :fieldLabel="__('engineerings::app.menu.scheduleFinish')" fieldRequired="true">
                </x-forms.label>
                <div class="bootstrap-timepicker timepicker">
                    <input type="datetime-local" id="schedule_finish" name="schedule_finish"
                        class="px-6 position-relative text-dark font-weight-normal form-control height-35 rounded p-0 text-left f-15">
                </div>
            </div>
            <div class="col-md-4">
                <div class="form-group mb-lg-0 mb-md-0 mb-4">
                    <x-forms.label class="mt-3" fieldId="estimate" :fieldLabel="__('engineerings::app.menu.estimateHours')"></x-forms.label>
                    <x-forms.input-group class="border px-2">
                        <label for="">@lang('engineerings::app.menu.hours'): </label>
                        <input type="text" name="estimate_hours" id="estimate_hours"
                            class="form-control-plaintext height-35 f-15 px-2 border-right" value="" readonly>
                        <label class="px-2">@lang('engineerings::app.menu.min'): </label>
                        <input type="text" name="estimate_minutes" id="estimate_minutes"
                            class="form-control-plaintext height-35 f-15 px-2" value="" readonly>
                    </x-forms.input-group>
                </div>
            </div>

            <div class="col-md-12">
                <div class="form-group">
                    <x-forms.label class="my-3" fieldId="description-text" :fieldLabel="__('engineerings::app.menu.workDesc')" fieldRequired="true">
                    </x-forms.label>
                    <textarea name="work_description" id="description-text" rows="5" class="form-control"></textarea>
                </div>
            </div>
            <hr class="m-0 border-top-grey">
            <div class="col-md-8">
                <div class="row">
                    <div class="col-md-4 mt-4">
                        <x-forms.label fieldId="rotation" :fieldLabel="__('modules.invoices.billingFrequency')" fieldRequired="true">
                        </x-forms.label>
                        <div class="form-group c-inv-select">
                            <select class="form-control select-picker" data-live-search="true" data-size="8"
                                name="rotation" id="rotation">
                                <option value="daily">@lang('app.daily')</option>
                                <option value="weekly">@lang('app.weekly')</option>
                                <option value="bi-weekly">@lang('app.bi-weekly')</option>
                                <option value="monthly">@lang('app.monthly')</option>
                                <option value="quarterly">@lang('app.quarterly')</option>
                                <option value="half-yearly">@lang('app.half-yearly')</option>
                                <option value="annually">@lang('app.annually')</option>
                            </select>
                        </div>
                    </div>
                    <div class="col-md-8 mt-4">
                        <div class="form-group">
                            <div class="d-flex">
                                <x-forms.label class="mr-3" fieldId="issue_date" :fieldLabel="__('app.startDate')">
                                </x-forms.label>
                                <x-forms.checkbox class="mr-0 mr-lg-2 mr-md-2 mt-0" :fieldLabel="__('Immediate start ( Schedule will generate from now )')"
                                    fieldName="immediate_schedule" fieldId="immediate_schedule" fieldValue="true"
                                    fieldRequired="true" />
                            </div>
                            <div class="input-group">
                                <input type="text" id="issue_date" name="issue_date"
                                    class="px-6 position-relative text-dark font-weight-normal form-control height-35 rounded p-0 text-left f-15"
                                    placeholder="@lang('placeholders.date')"
                                    value="{{ Carbon\Carbon::now(company()->timezone)->format(company()->date_format) }}">
                            </div>
                            <small class="form-text text-muted">@lang('Date from which schedule will be created')</small>
                        </div>
                    </div>
                    <div class="col-lg-4 mt-0">
                        <x-forms.number class="mr-0 mr-lg-2 mr-md-2 mt-0" :fieldLabel="__('modules.invoices.totalCount')" fieldName="billing_cycle"
                            fieldId="billing_cycle" :fieldHelp="__('No. of schedule cycles to be charged (set -1 for infinite cycles)')" fieldRequired="true" />
                    </div>
                </div>
            </div>

            <div class="col-md-4 mt-4 information-box">
                <p id="plan">@lang('Schedule will be generate Daily')</p>
                <p id="current_date">@lang('First Schedule will be generated on')
                    {{ Carbon\Carbon::now()->translatedFormat(company()->date_format) }}</p>
                <p id="next_date">@lang('Next Schedule Date will be')
                    {{ Carbon\Carbon::now()->addDay()->translatedFormat(company()->date_format) }}</p>
                <p>@lang('And so on....')</p>
                <span id="billing"></span>
            </div>

        </div>

        <!-- CANCEL SAVE START -->
        <x-form-actions class="c-inv-btns d-block d-lg-flex d-md-flex">
            <x-forms.button-primary data-type="save" class="save-form mr-3" icon="check">@lang('app.save')
            </x-forms.button-primary>

            <x-forms.button-cancel :link="route('recurring-work.index')" class="border-0 ">@lang('app.cancel')
            </x-forms.button-cancel>
        </x-form-actions>
        <!-- CANCEL SAVE END -->

    </x-form>
    <!-- FORM END -->
</div>
<script>
    $(document).ready(function() {

        if ($('.custom-date-picker').length > 0) {
            datepicker('.custom-date-picker', {
                position: 'bl',
                ...datepickerConfig
            });
        }

        const dp1 = datepicker('#issue_date', {
            position: 'bl',
            onSelect: (instance, date) => {
                var rotation = $('#rotation').val();
                nextDate(rotation, date);
            },
            ...datepickerConfig
        });

        $('#unit_id').change(function(e) {
            getAssets()
        });

        function getAssets() {
            var unit_id = document.getElementById("unit_id").value;
            var url = "{{ route('work.get_assets', ':id') }}";
            url = (unit_id) ? url.replace(':id', unit_id) : url.replace(':id', null);
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
                                .type.type_name + '</option>';
                            options.push(selectData);
                        });

                        $('#assets_id').html('<option value="">--</option>' +
                            options);
                        $('#assets_id').selectpicker('refresh');
                    }
                }
            })
        }

        // Ambil elemen input untuk schedule_start dan schedule_finish
        var startInput = document.getElementById("schedule_start");
        var finishInput = document.getElementById("schedule_finish");

        // Ambil elemen input untuk estimate_hours dan estimate_minutes
        var jamInput = document.getElementById("estimate_hours");
        var minInput = document.getElementById("estimate_minutes");

        // Tambahkan event listener pada kedua elemen input
        startInput.addEventListener("input", updateEstimate);
        finishInput.addEventListener("input", updateEstimate);

        // Fungsi untuk menghitung selisih waktu dan menampilkan hasilnya pada input 
        function updateEstimate() {
            var start = new Date(startInput.value);
            var finish = new Date(finishInput.value);
            var diff = finish - start;
            var hours = diff / (1000 * 60 * 60);
            var jam = Math.floor(hours);
            var min = Math.floor((hours % 1) * 60);

            // Tampilkan hasil pada input estimate_hours dan estimate_minutes
            jamInput.value = jam;
            minInput.value = min;
        }

        function ucWord(str) {
            str = str.toLowerCase().replace(/\b[a-z]/g, function(letter) {
                return letter.toUpperCase();
            });
            return str;
        }

        $('.save-form').click(function() {
            $.easyAjax({
                url: "{{ route('recurring-work.store') }}",
                container: '#saveInvoiceForm',
                type: "POST",
                blockUI: true,
                redirect: true,
                data: $('#saveInvoiceForm').serialize(),
                success: function(response) {
                    if (response.status === 'success') {
                        window.location.href = response.redirectUrl;
                    }
                }
            });
        });

        init(RIGHT_MODAL);
    });

    $('body').on('change keyup', '#rotation, #billing_cycle', function() {
        var billingCycle = $('#billing_cycle').val();
        billingCycle != '' ? $('#billing').html("{{ __('No. of billing cycle is') }}" + ' ' + billingCycle) :
            $(
                '#billing').html('');

        var rotation = $('#rotation').val();
        switch (rotation) {
            case 'daily':
                var rotationType = "{{ __('app.daily') }}";
                break;
            case 'weekly':
                var rotationType = "{{ __('app.every') }}" + ' ' + "{{ __('Week') }}";
                break;
            case 'bi-weekly':
                var rotationType = "{{ __('app.every') }}" + ' ' + "{{ __('Bi-Week') }}";
                break;
            case 'monthly':
                var rotationType = "{{ __('app.every') }}" + ' ' + "{{ __('Month') }}";
                break;
            case 'quarterly':
                var rotationType = "{{ __('app.every') }}" + ' ' + "{{ __('Quarter') }}";
                break;
            case 'half-yearly':
                var rotationType = "{{ __('app.every') }}" + ' ' + "{{ __('Half Year') }}";
                break;
            case 'annually':
                var rotationType = "{{ __('app.every') }}" + ' ' + "{{ __('Year') }}";
                break;
            default:
        }

        $('#plan').html("{{ __('Schedule will be generate') }}" + ' ' + rotationType);
        if ($('#immediate_schedule').is(':checked')) {
            var date = moment().toDate();
        } else {
            var startDate = $('#issue_date').val();
            var date = moment(startDate, 'DD-MM-YYYY').toDate();
        }

        nextDate(rotation, date);
    })

    $('#immediate_schedule').change(function() {
        var rotation = $('#rotation').val();
        if ($(this).is(':checked')) {
            var date = moment().toDate();
            $('#issue_date').val(moment(date, "DD-MM-YYYY").format('{{ company()->moment_date_format }}'));
            $('#issue_date').prop('disabled', true)
        } else {
            $('#issue_date').prop('disabled', false)
            var startDate = $('#issue_date').val();
            var date = moment(startDate, 'DD-MM-YYYY').toDate();
        }
        nextDate(rotation, date);
    })

    function nextDate(rotation, date) {
        var nextDate = moment(date, "DD-MM-YYYY");
        var currentValue = nextDate.format('{{ company()->moment_date_format }}');

        switch (rotation) {
            case 'daily':
                var rotationDate = nextDate.add(1, 'days');
                break;
            case 'weekly':
                var rotationDate = nextDate.add(1, 'weeks');
                break;
            case 'bi-weekly':
                var rotationDate = nextDate.add(2, 'weeks');
                break;
            case 'monthly':
                var rotationDate = nextDate.add(1, 'months');
                break;
            case 'quarterly':
                var rotationDate = nextDate.add(1, 'quarters');
                break;
            case 'half-yearly':
                var rotationDate = nextDate.add(2, 'quarters');
                break;
            case 'annually':
                var rotationDate = nextDate.add(1, 'years');
                break;
            default:
        }
        var value = rotationDate.format('{{ company()->moment_date_format }}');

        $('#current_date').html("{{ __('First Schedule will be generated on') }}" + ' ' + currentValue);
        $('#next_date').html("{{ __('Next Schedule Date will be') }}" + ' ' + value);
    }
</script>
