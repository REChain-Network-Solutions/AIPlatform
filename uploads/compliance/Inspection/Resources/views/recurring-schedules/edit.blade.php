@extends('layouts.app')

@section('content')

    @php
        $billingCycle = $schedule->unlimited_recurring == 1 ? -1 : $schedule->billing_cycle;
    @endphp

    @php
        $recurringSchedule = count($schedule->recurrings) > 0 ? 'disabled' : '';
    @endphp
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
            <x-form class="c-inv-form" id="saveInvoiceForm"> @method("PUT")
            <input type="hidden" name="schedule_count" value="{{count($schedule->recurrings)}}">
                <!-- INVOICE NUMBER, DATE, DUE DATE, FREQUENCY START -->
                <div class="row px-lg-4 px-md-4 px-3 pt-3">
                    <!-- BILLING FREQUENCY -->
                    <div class="col-md-3">
                        <div class="form-group c-inv-select mb-4">
                            <x-forms.text fieldId="subject" :fieldLabel="__('Schedule Name')"
                            :fieldValue="$schedule->subject" fieldName="subject" fieldRequired="true" fieldPlaceholder="e.g. Toilet 1st floor etc." :fieldReadOnly="(count($schedule->recurrings) > 0) ? true : ''">
                            </x-forms.text>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <x-forms.label class="mt-3" fieldId="parent_label" :fieldLabel="__('units::app.menu.tower')"
                                    fieldName="parent_label">
                        </x-forms.label>
                        <x-forms.input-group>
                            <select class="form-control select-picker" name="tower_id" id="tower_id" {{$recurringSchedule}}
                                    data-live-search="true">
                                <option value="">--</option>
                                @foreach ($towers as $tower)
                                <option @if ($tower->id == $schedule->tower_id) selected
                                        @endif value="{{ $tower->id }}">
                                    {{ $tower->tower_name }}
                                </option>
                                @endforeach
                            </select>
                        </x-forms.input-group>
                    </div>
                    <div class="col-md-3">
                        <x-forms.label class="mt-3" fieldId="parent_label" :fieldLabel="__('units::app.menu.floor')"
                                    fieldName="parent_label">
                        </x-forms.label>
                        <x-forms.input-group>
                            <select class="form-control select-picker" name="floor_id" id="floor_id" {{$recurringSchedule}}
                                    data-live-search="true">
                                <option value="">--</option>
                                @foreach ($floors as $floor)
                                <option @if ($floor->id == $schedule->floor_id) selected
                                        @endif value="{{ $floor->id }}">
                                    {{ $floor->floor_name }}
                                </option>
                                @endforeach
                            </select>
                        </x-forms.input-group>
                    </div>
                    <div class="col-md-3">
                        <div class="form-group c-inv-select mb-4">
                            <x-forms.label fieldId="status" :fieldLabel="__('app.status')">
                            </x-forms.label>
                            <div class="select-others height-35 rounded">
                                <select class="form-control select-picker" name="status" id="status">
                                    <option @if ($schedule->status == 'active') selected
                                            @endif value="active">@lang('app.active')
                                    </option>
                                    <option @if ($schedule->status == 'inactive') selected
                                            @endif value="inactive">@lang('app.inactive')
                                    </option>
                                </select>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <x-forms.text fieldId="lokasi" :fieldLabel="__('Location')"
                        :fieldValue="$schedule->lokasi" fieldName="lokasi" fieldRequired="true" fieldPlaceholder="" :fieldReadOnly="(count($schedule->recurrings) > 0) ? true : ''">
                        </x-forms.text>
                    </div>
                    <div class="col-md-2">
                        <x-forms.text fieldId="shift" :fieldLabel="__('Shift')"
                        :fieldValue="$schedule->shift" fieldName="shift" fieldRequired="true" fieldPlaceholder="" :fieldReadOnly="(count($schedule->recurrings) > 0) ? true : ''">
                        </x-forms.text>
                    </div>

                    <div class="col-md-3 col-lg-3">
                        <div class="bootstrap-timepicker timepicker">
                            <x-forms.text :fieldLabel="__('Start Time')"
                                :fieldPlaceholder="__('placeholders.hours')" fieldName="awal"
                                :fieldValue="$schedule->awal" fieldId="awal" fieldRequired="true" />
                        </div>
                    </div>

                    <div class="col-md-3 col-lg-3">
                        <div class="bootstrap-timepicker timepicker">
                            <x-forms.text :fieldLabel="__('Finish Time')"
                                :fieldPlaceholder="__('placeholders.hours')" fieldName="akhir"
                                :fieldValue="$schedule->akhir" fieldId="akhir" fieldRequired="true" />
                        </div>
                    </div>
                </div>
                <hr class="m-0 border-top-grey">
                <div class="row px-lg-4 px-md-4 px-3 pt-3">
                    <div class="col-md-8">
                        <div class="row">
                            <!-- BILLING FREQUENCY -->
                            <div class="col-md-4 mt-4">
                                <div class="form-group c-inv-select mb-4">
                                    <x-forms.label fieldId="rotation" :fieldLabel="__('modules.inspection_schedules.billingFrequency')"
                                                fieldRequired="true">
                                    </x-forms.label>
                                    <select class="form-control select-picker" data-live-search="true" data-size="8"
                                            name="rotation"
                                            id="rotation" {{$recurringSchedule}}>
                                        <option value="daily" @if($schedule->rotation == 'daily') selected @endif>@lang('app.daily')</option>
                                        <option value="weekly" @if($schedule->rotation == 'weekly') selected @endif>@lang('app.weekly')</option>
                                        <option value="bi-weekly" @if($schedule->rotation == 'bi-weekly') selected @endif>@lang('app.bi-weekly')</option>
                                        <option value="monthly" @if($schedule->rotation == 'monthly') selected @endif>@lang('app.monthly')</option>
                                        <option value="quarterly" @if($schedule->rotation == 'quarterly') selected @endif>@lang('app.quarterly')</option>
                                        <option value="half-yearly" @if($schedule->rotation == 'half-yearly') selected @endif>@lang('app.half-yearly')</option>
                                        <option value="annually" @if($schedule->rotation == 'annually') selected @endif>@lang('app.annually')</option>
                                    </select>
                                </div>
                            </div>
                            <!-- BILLING FREQUENCY -->
                            <div class="col-md-8 mt-4">
                                <div class="form-group mb-lg-0 mb-md-0 mb-4">
                                    <x-forms.label fieldId="issue_date" :fieldLabel="__('app.startDate')">
                                    </x-forms.label>
                                    <input type="text" id="issue_date" name="issue_date"
                                        class="px-6 position-relative text-dark font-weight-normal form-control height-35 rounded p-0 text-left f-15"
                                        placeholder="@lang('placeholders.date')"
                                        value="{{ $schedule->issue_date->translatedFormat(company()->date_format) }}" {{$recurringSchedule}}>

                                    <small class="form-text text-muted">@lang('modules.recurringSchedule.scheduleDate')</small>
                                </div>
                            </div>
                            <div class="col-lg-4 mt-0 billingInterval">
                                <x-forms.number class="mr-0  mr-lg-2 mr-md-2 mt-0"
                                                :fieldLabel="__('modules.inspection_schedules.totalCount')"
                                                fieldName="billing_cycle" fieldId="billing_cycle" :fieldValue="$billingCycle"
                                                :fieldHelp="__('modules.inspection_schedules.noOfBillingCycle')" :fieldReadOnly="(count($schedule->recurrings) > 0) ? true : ''"/>
                            </div>
                        </div>
                    </div>
                    @php
                        switch ($schedule->rotation) {
                            case 'daily':
                                $rotationType = __('app.daily');
                                break;
                            case 'weekly':
                                $rotationType = __('modules.recurringSchedule.week');
                                break;
                            case 'bi-weekly':
                                $rotationType = __('app.bi-week');
                                break;
                            case 'monthly':
                                $rotationType = __('app.month');
                                break;
                            case 'quarterly':
                                $rotationType = __('app.quarter');
                                break;
                            case 'half-yearly':
                                $rotationType = __('app.half-year');
                                break;
                            case 'annually':
                                $rotationType = __('app.year');
                                break;
                            default:
                        }
                    @endphp
                    <div class="col-md-4 mt-4 information-box">
                        <p id="plan">@lang('modules.inspection_schedules.customerCharged') @if($schedule->rotation != 'daily') @lang('app.every') @endif {{$rotationType}}</p>
                        @if (count($schedule->recurrings) == 0)
                            <p id="current_date">@lang('modules.recurringSchedule.currentScheduleDate') {{$schedule->issue_date->translatedFormat(company()->date_format)}}</p>
                        @endif
                        <p id="next_date"></p>
                        @if (count($schedule->recurrings) == 0)
                            <p>@lang('modules.recurringSchedule.soOn')</p>
                        @endif
                        <p id="billing">@lang('modules.recurringSchedule.billingCycle') {{$billingCycle}}</p>
                        <input type="hidden" id="next_schedule" value="{{ $schedule->issue_date->translatedFormat(company()->date_format) }}">
                    </div>
                </div>

                <hr class="m-0 border-top-grey">

                <div class="px-lg-4 px-md-4 px-3 py-3">
                    <h4 class="mb-0 f-21 font-weight-normal text-capitalize">Standar Bersih</h4>
                </div>
                <div id="sortable">
                @foreach ($schedule->items as $key => $item)
                    <!-- DESKTOP DESCRIPTION TABLE START -->
                        <div class="d-flex px-4 py-3 c-inv-desc item-row">
                            <div class="c-inv-desc-table w-100 d-lg-flex d-md-flex d-block">

                                <input type="hidden" name="item_ids[]" value="{{ $item->id }}">
                                <input type="text" class="f-14 border-0 w-100 item_name" name="item_name[]"
                                        placeholder="@lang('modules.expenses.itemName')"
                                        value="{{ $item->item_name }}" {{$recurringSchedule}}>

                                @if(count($schedule->recurrings) == 0)
                                <a href="javascript:;"
                                class="d-flex align-items-center justify-content-center ml-3 remove-item"><i
                                        class="fa fa-times-circle f-20 text-lightest"></i></a>
                                @endif
                            </div>
                        </div>
                        <!-- DESKTOP DESCRIPTION TABLE END -->
                    @endforeach
                </div>
                <!--  ADD ITEM START-->
                    <div class="row px-lg-4 px-md-4 px-3 pb-3 pt-0 mb-3  mt-2">
                        <div class="col-md-12">
                            <a class="f-15 f-w-500" href="javascript:;" id="add-item"><i
                                    class="icons icon-plus font-weight-bold mr-1"></i>@lang('modules.inspection_schedules.addItem')</a>
                        </div>
                    </div>
                <!--  ADD ITEM END-->

                <hr class="m-0 border-top-grey">

                <!-- CANCEL SAVE SEND START -->
                <div class="px-lg-4 px-md-4 px-3 py-3 c-inv-btns">
                    <div class="d-flex">
                        <x-forms.button-cancel :link="route('recurring-inspection_schedules.index')"
                                            class="border-0 mr-2">@lang('app.cancel')
                        </x-forms.button-cancel>
                        <x-forms.button-primary class="save-form"
                                                icon="check">@lang('app.save')</x-forms.button-primary>
                    </div>
                </div>
                <!-- CANCEL SAVE SEND END -->

            </x-form>
            <!-- FORM END -->
        </div>
        <!-- CREATE INVOICE END -->
    </div>

@endsection

@push('scripts')
    <script>
        $(document).ready(function () {
            var schedule = @json($schedule);
            var rotation = @json($schedule->rotation);
            var startDate =$('#next_schedule').val();
            var date = moment(startDate, 'DD-MM-YYYY').toDate();
            nextDate(rotation, date)




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
                dateSelected: new Date("{{ str_replace('-', '/', $schedule->issue_date) }}"),
                ...datepickerConfig
            });

            $('#awal, #akhir').timepicker({
                @if (company()->time_format == 'H:i')
                    showMeridian: false,
                @endif
            }).on('hide.timepicker', function(e) {
                calculateTime();
            });

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


            $('#saveInvoiceForm').on('click', '.remove-item', function () {
                $(this).closest('.item-row').fadeOut(300, function () {
                    $(this).remove();
                });
            });

            $('.save-form').click(function () {
                $.easyAjax({
                    url: "{{ route('recurring-inspection_schedules.update', $schedule->id) }}",
                    container: '#saveInvoiceForm',
                    type: "POST",
                    blockUI: true,
                    file: true,
                    data: $('#saveInvoiceForm').serialize(),
                        success: function(response) {
                            if (response.status === 'success') {

                                    window.location.href = response.redirectUrl;

                            }
                        }
                })


            });



            $('body').on('change keyup', '#rotation, #billing_cycle', function(){
                var billingCycle = $('#billing_cycle').val();
                billingCycle != '' ? $('#billing').html("{{__('modules.recurringSchedule.billingCycle')}}" +' '+billingCycle) : $('#billing').html('');

                var rotation = $('#rotation').val();

                switch (rotation) {
                case 'daily':
                    var rotationType = "{{__('app.daily')}}";
                    break;
                case 'weekly':
                    var rotationType = "{{__('app.every')}}"+' '+"{{__('modules.recurringSchedule.week')}}";
                    break;
                case 'bi-weekly':
                    var rotationType = "{{__('app.every')}}"+' '+"{{__('app.bi-week')}}";
                    break;
                case 'monthly':
                    var rotationType = "{{__('app.every')}}"+' '+"{{__('app.month')}}";
                    break;
                case 'quarterly':
                    var rotationType = "{{__('app.every')}}"+' '+"{{__('app.quarter')}}";
                    break;
                case 'half-yearly':
                    var rotationType = "{{__('app.every')}}"+' '+"{{__('app.half-year')}}";
                    break;
                case 'annually':
                    var rotationType = "{{__('app.every')}}"+' '+"{{__('app.year')}}";
                    break;
                default:
                //
                }

                $('#plan').html("{{__('modules.inspection_schedules.customerCharged')}}" + ' ' + rotationType);

                var startDate = $('#issue_date').val();
                var date = moment(startDate, 'DD-MM-YYYY').toDate();

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

                $('#current_date').html("{{__('modules.recurringSchedule.currentScheduleDate')}}" + ' ' + currentValue);

                $('#next_date').html("{{__('modules.recurringSchedule.nextScheduleDate')}}" + ' ' + value);
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
        });
    </script>
@endpush
