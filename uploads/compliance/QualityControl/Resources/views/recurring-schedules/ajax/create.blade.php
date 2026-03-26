<style>
    .customSequence .btn {
        border: none;
    }

    .information-box {
        border-style: dotted;
        margin-bottom: 30px;
        margin-top:10px;
        padding-top: 10px;
        border-radius: 4px;
    }
</style>
<div class="content-wrapper">
<!-- CREATE INVOICE START -->
<div class="bg-white rounded b-shadow-4 create-inv">
    <!-- HEADING START -->
    <div class="px-lg-4 px-md-4 px-3 py-3">
        <h4 class="mb-0 f-21 font-weight-normal text-capitalize">@lang('app.schedule') @lang('app.details')</h4>
    </div>
    <!-- HEADING END -->
    <hr class="m-0 border-top-grey">
    <!-- FORM START -->
    <x-form class="c-inv-form" id="saveScheduleForm">
        <hr class="m-0 border-top-grey">
        <div class="row px-lg-4 px-md-4 px-3 pt-3">
            <!-- BILLING FREQUENCY -->
            <div class="col-md-12">
                <div class="row">
                    <div class="col-lg-6">
                        <x-forms.text fieldId="subject" :fieldLabel="__('Schedule Name')"
                            fieldName="subject" fieldRequired="true" fieldPlaceholder="e.g. Toilet 1st floor etc.">
                        </x-forms.text>
                    </div>

                    <div class="col-md-3">
                        <x-forms.label class="mt-3" fieldId="parent_label" :fieldLabel="__('units::app.menu.tower')"
                                       fieldName="parent_label">
                        </x-forms.label>
                        <x-forms.input-group>
                            <select class="form-control select-picker" name="tower_id" id="tower_id"
                                    data-live-search="true">
                                <option value="">--</option>
                                @foreach($towers as $tower)
                                    <option value="{{ $tower->id }}">{{ $tower->tower_name }}</option>
                                @endforeach
                            </select>
                        </x-forms.input-group>
                    </div>
                    <div class="col-md-3">
                        <x-forms.label class="mt-3" fieldId="parent_label" :fieldLabel="__('units::app.menu.floor')"
                                       fieldName="parent_label">
                        </x-forms.label>
                        <x-forms.input-group>
                            <select class="form-control select-picker" name="floor_id" id="floor_id"
                                    data-live-search="true">
                                <option value="">--</option>
                                @foreach($floors as $floor)
                                    <option value="{{ $floor->id }}">{{ $floor->floor_name }}</option>
                                @endforeach
                            </select>
                        </x-forms.input-group>
                    </div>
                </div>
                <div class="row">
                    <div class="col-md-4">
                        <x-forms.text fieldId="lokasi" :fieldLabel="__('Location')"
                            fieldName="lokasi" fieldRequired="true" fieldPlaceholder="">
                        </x-forms.text>
                    </div>
                    <div class="col-md-2">
                        <x-forms.text fieldId="shift" :fieldLabel="__('Shift')"
                            fieldName="shift" fieldRequired="true" fieldPlaceholder="">
                        </x-forms.text>
                    </div>

                    <div class="col-md-3 col-lg-3">
                        <div class="bootstrap-timepicker timepicker">
                            <x-forms.text :fieldLabel="__('Start Time')"
                                :fieldPlaceholder="__('placeholders.hours')" fieldName="awal"
                                fieldId="awal" fieldRequired="true" />
                        </div>
                    </div>

                    <div class="col-md-3 col-lg-3">
                        <div class="bootstrap-timepicker timepicker">
                            <x-forms.text :fieldLabel="__('Finish Time')"
                                :fieldPlaceholder="__('placeholders.hours')" fieldName="akhir"
                                fieldId="akhir" fieldRequired="true" />
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-md-8">
                <div class="row">
                    <div class="col-md-4 mt-4">
                        <x-forms.label fieldId="rotation" :fieldLabel="__('Schedule Frequency')"
                                    fieldRequired="true">
                        </x-forms.label>
                        <div class="form-group c-inv-select">
                            <select class="form-control select-picker" data-live-search="true" data-size="8" name="rotation"
                                    id="rotation">
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
                                <x-forms.checkbox class="mr-0 mr-lg-2 mr-md-2 mt-0"
                                                :fieldLabel="__('Immediate start ( Schedule will generate from now )')"
                                                fieldName="immediate_schedule"
                                                fieldId="immediate_schedule" fieldValue="true" fieldRequired="true"/>
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
                        <x-forms.number class="mr-0 mr-lg-2 mr-md-2 mt-0" :fieldLabel="__('Total Count')"
                                        fieldName="billing_cycle" fieldId="billing_cycle"
                                        :fieldHelp="__('No. of schedule cycles to be charged (set -1 for infinite cycles)')" fieldRequired="true"/>
                    </div>
                </div>
            </div>

            <div class="col-md-4 mt-4 information-box">
                <p id="plan">@lang('Schedule will be generate Daily')</p>
                <p id="current_date">@lang('First Schedule will be generated on') {{Carbon\Carbon::now()->translatedFormat(company()->date_format)}}</p>
                <p id="next_date">@lang('Next Schedule Date will be') {{Carbon\Carbon::now()->addDay()->translatedFormat(company()->date_format)}}</p>
                <p>@lang('And so on....')</p>
                <span id="billing"></span>
            </div>

        </div>

        <hr class="m-0 border-top-grey">


        <div class="px-lg-4 px-md-4 px-3 py-3">
            <h4 class="mb-0 f-21 font-weight-normal text-capitalize">Standar Bersih</h4>
        </div>
        <div id="sortable">

            <!-- DESKTOP DESCRIPTION TABLE START -->

                <div class="d-flex px-4 py-3 c-inv-desc item-row">
                    <div class="c-inv-desc-table w-100 d-lg-flex d-md-flex d-block">

                        <input type="text" class="form-control f-14 border-0 w-100 item_name"
                                           name="item_name[]" placeholder="">

                        <a href="javascript:;"
                           class="d-flex align-items-center justify-content-center ml-3 remove-item"><i
                                class="fa fa-times-circle f-20 text-lightest"></i></a>
                    </div>
                </div>
            <!-- DESKTOP DESCRIPTION TABLE END -->
        </div>
        <!--  ADD ITEM START-->
        <div class="row px-lg-4 px-md-4 px-3 pb-3 pt-0 mb-3  mt-2">
            <div class="col-md-12">
                <a class="f-15 f-w-500" href="javascript:;" id="add-item"><i
                        class="icons icon-plus font-weight-bold mr-1"></i>@lang('Add Item')</a>
            </div>
        </div>
        <!--  ADD ITEM END-->

        <!-- CANCEL SAVE SEND START -->
        <div class="px-lg-4 px-md-4 px-3 py-3 c-inv-btns">

            <x-forms.button-cancel :link="route('recurring-inspection_schedules.index')" class="border-0 mr-3">@lang('app.cancel')
            </x-forms.button-cancel>

            <x-forms.button-primary id="save-form" icon="check">@lang('app.save')</x-forms.button-primary>

        </div>
        <!-- CANCEL SAVE SEND END -->

    </x-form>
    <!-- FORM END -->
</div>
<!-- CREATE INVOICE END -->
</div>
<script>
    $(document).ready(function () {

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

        $('#awal, #akhir').timepicker({
            @if (company()->time_format == 'H:i')
                showMeridian: false,
            @endif
        }).on('hide.timepicker', function(e) {
            calculateTime();
        });

        function ucWord(str){
            str = str.toLowerCase().replace(/\b[a-z]/g, function(letter) {
                return letter.toUpperCase();
            });
            return str;
        }

        $(document).on('click', '#add-item', function () {

            var i = $(document).find('.item_name').length;
            var item = ' <div class="d-flex px-4 py-3 c-inv-desc item-row">' +
                '<div class="c-inv-desc-table w-100 d-lg-flex d-md-flex d-block">' +

                '<input type="text" class="form-control f-14 border-0 w-100 item_name" name="item_name[]" >' +

                '<a href="javascript:;" class="d-flex align-items-center justify-content-center ml-3 remove-item"><i class="fa fa-times-circle f-20 text-lightest"></i></a>' +
                '</div>' +
                '</div>';

            $(item).hide().appendTo("#sortable").fadeIn(500);
            $('#multiselect' + i).selectpicker();

            $('#dropify' + i).dropify({
                messages: dropifyMessages
            });
        });

        $('#saveScheduleForm').on('click', '.remove-item', function () {
            $(this).closest('.item-row').fadeOut(300, function () {
                $(this).remove();
            });
        });

        $('#save-form').click(function () {
            if (KTUtil.isMobileDevice()) {
                $('.desktop-description').remove();
            } else {
                $('.mobile-description').remove();
            }

            $.easyAjax({
                url: "{{ route('recurring-inspection_schedules.store') }}" ,
                container: '#saveScheduleForm',
                type: "POST",
                blockUI: true,
                redirect: true,
                disableButton: true,
                file: true,
                data: $('#saveScheduleForm').serialize()
            })
        });

        init(RIGHT_MODAL);
    });



    $('body').on('change keyup', '#rotation, #billing_cycle', function () {
        var billingCycle = $('#billing_cycle').val();
        billingCycle != '' ? $('#billing').html("{{__('No. of billing cycle is')}}" + ' ' + billingCycle) : $('#billing').html('');

        var rotation = $('#rotation').val();
        switch (rotation) {
            case 'daily':
                var rotationType = "{{__('app.daily')}}";
                break;
            case 'weekly':
                var rotationType = "{{__('app.every')}}"+' '+"{{__('Week')}}";
                break;
            case 'bi-weekly':
                var rotationType = "{{__('app.every')}}"+' '+"{{__('Bi-Week')}}";
                break;
            case 'monthly':
                var rotationType = "{{__('app.every')}}"+' '+"{{__('Month')}}";
                break;
            case 'quarterly':
                var rotationType = "{{__('app.every')}}"+' '+"{{__('Quarter')}}";
                break;
            case 'half-yearly':
                var rotationType = "{{__('app.every')}}"+' '+"{{__('Half Year')}}";
                break;
            case 'annually':
                var rotationType = "{{__('app.every')}}"+' '+"{{__('Year')}}";
                break;
            default:
            //
        }

        $('#plan').html("{{__('Schedule will be generate')}}" + ' ' + rotationType);

        if ($('#immediate_schedule').is(':checked')) {
            var date = moment().toDate();
        } else {
            var startDate = $('#issue_date').val();
            var date = moment(startDate, 'DD-MM-YYYY').toDate();
        }

        nextDate(rotation, date);
    })

    $('#immediate_schedule').change(function () {
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
            //
        }

        var value = rotationDate.format('{{ company()->moment_date_format }}');

        $('#current_date').html("{{__('First Schedule will be generated on')}}" + ' ' + currentValue);

        $('#next_date').html("{{__('Next Schedule Date will be')}}" + ' ' + value);
    }

    function calculateTime() {
        var format = '{{ company()->moment_date_format }}';
        var startDate = $('#issue_date').val();
        var endDate = $('#issue_date').val();
        var startTime = $("#awal").val();
        var endTime = $("#akhir").val();

        startDate = moment(startDate, format).format('YYYY-MM-DD');
        endDate = moment(endDate, format).format('YYYY-MM-DD');

        var timeStart = new Date(startDate + " " + startTime);
        var timeEnd = new Date(endDate + " " + endTime);

        var diff = (timeEnd - timeStart) / 60000; //dividing by seconds and milliseconds

        var minutes = diff % 60;
        var hours = (diff - minutes) / 60;

        if (hours < 0 || minutes < 0) {
            Swal.fire({
                icon: 'warning',
                text: "@lang('messages.totalTimeZero')",

                customClass: {
                    confirmButton: 'btn btn-primary',
                },
                showClass: {
                    popup: 'swal2-noanimation',
                    backdrop: 'swal2-noanimation'
                },
                buttonsStyling: false
            });
            $("#awal").val(startTime);
            $('#akhir').val(endTime);

            return false;

            calculateTime();
        } else {

        }

    }
</script>
