@php

@endphp

<!-- CREATE INVOICE START -->
<div class="bg-white rounded b-shadow-4 create-inv">
    <!-- HEADING START -->
    <div class="px-lg-4 px-md-4 px-3 py-3">
        <h4 class="mb-0 f-21 font-weight-normal text-capitalize">Schedule @lang('app.details')</h4>
    </div>
    <!-- HEADING END -->
    <hr class="m-0 border-top-grey">
    <!-- FORM START -->
    <x-form class="c-inv-form" id="saveScheduleForm">
        <!-- INVOICE NUMBER, DATE, DUE DATE, FREQUENCY START -->
        <div class="row px-lg-4 px-md-4 px-3 py-3">
            <!-- INVOICE NUMBER START -->
            <div class="col-md-4">
                <div class="form-group mb-lg-0 mb-md-0 mb-4">
                    <x-forms.text fieldId="subject" :fieldLabel="__('Schedule Job')"
                            fieldName="subject" fieldRequired="true" fieldPlaceholder="e.g. Toilet 1st floor etc.">
                    </x-forms.text>
                </div>
            </div>
            <div class="col-md-2">
                <div class="form-group mb-lg-0 mb-md-0 mb-4">
                    <x-forms.label class="mt-3" fieldId="issue_date" :fieldLabel="__('app.startDate')">
                    </x-forms.label>
                    <input type="text" id="issue_date" name="issue_date"
                        class="px-6 position-relative text-dark font-weight-normal form-control height-35 rounded p-0 text-left f-15"
                        placeholder="@lang('placeholders.date')"
                        value="{{ Carbon\Carbon::now(company()->timezone)->format(company()->date_format) }}">
                </div>
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
        <!-- INVOICE NUMBER, DATE, DUE DATE, FREQUENCY END -->

        <hr class="m-0 border-top-grey">

        <!-- CLIENT, PROJECT, GST, BILLING, SHIPPING ADDRESS START -->
        <div class="row px-lg-4 px-md-4 px-3 pt-3">
            <div class="col-md-4">
                <x-forms.text fieldId="lokasi" :fieldLabel="__('Location')"
                    fieldName="lokasi" fieldRequired="true" fieldPlaceholder="">
                </x-forms.text>
            </div>
            <div class="col-md-1">
                <x-forms.text fieldId="shift" :fieldLabel="__('Shift')"
                    fieldName="shift" fieldRequired="true" fieldPlaceholder="">
                </x-forms.text>
            </div>

            <div class="col-md-2">
                <div class="bootstrap-timepicker timepicker">
                    <x-forms.text :fieldLabel="__('Start Time')"
                        :fieldPlaceholder="__('placeholders.hours')" fieldName="awal"
                        fieldId="awal" fieldRequired="true" />
                </div>
            </div>

            <div class="col-md-2">
                <div class="bootstrap-timepicker timepicker">
                    <x-forms.text :fieldLabel="__('Finish Time')"
                        :fieldPlaceholder="__('placeholders.hours')" fieldName="akhir"
                        fieldId="akhir" fieldRequired="true" />
                </div>
            </div>
            <div class="col-md-2 col-lg-3">
                <x-forms.select fieldId="worker_id" :fieldLabel="'Choose Worker'" fieldName="worker_id" search="true"
                    fieldRequired="true">
                    <option value="">--</option>
                    @foreach ($employees as $employee)
                        <x-user-option :user="$employee" :selected="request()->has('default_assign') &&
                        request('default_assign') == $employee->id" />
                    @endforeach
                </x-forms.select>
            </div>
        </div>

        <!-- CLIENT, PROJECT, GST, BILLING, SHIPPING ADDRESS END -->



        <hr class="m-0 border-top-grey">
        <div class="px-lg-4 px-md-4 px-3 py-3">
            <h3 class="mb-0 f-21 font-weight-normal text-capitalize">Standar Bersih</h4>
        </div>

        <div id="sortable">
            @if (isset($schedule))
                @foreach ($schedule->items as $key => $item)
                    <!-- DESKTOP DESCRIPTION TABLE START -->
                    <div class="d-flex px-4 py-3 c-inv-desc item-row">
                        <div class="c-inv-desc-table w-100 d-lg-flex d-md-flex d-block">
                            <input type="text" class="form-control f-14 border-0 w-100 item_name"
                                name="item_name[]" placeholder="@lang('modules.expenses.itemName')"
                                value="{{ $item->item_name }}">
                            <a href="javascript:;"
                                class="d-flex align-items-center justify-content-center ml-3 remove-item"><i
                                    class="fa fa-times-circle f-20 text-lightest"></i></a>
                        </div>
                    </div>
                    <!-- DESKTOP DESCRIPTION TABLE END -->
                @endforeach
            @else
                <!-- DESKTOP DESCRIPTION TABLE START -->
                <div class="d-flex px-4 py-3 c-inv-desc item-row">
                    <div class="c-inv-desc-table w-100 d-lg-flex d-md-flex d-block">
                        <input type="text" class="form-control f-14 border-0 w-100 item_name"
                            name="item_name[]" placeholder="@lang('modules.expenses.itemName')">
                        <a href="javascript:;"
                            class="d-flex align-items-center justify-content-center ml-3 remove-item"><i
                                class="fa fa-times-circle f-20 text-lightest"></i></a>
                    </div>
                </div>
                <!-- DESKTOP DESCRIPTION TABLE END -->
            @endif

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

        <!-- TOTAL, DISCOUNT START -->

        <!-- TOTAL, DISCOUNT END -->

        <!-- NOTE AND TERMS AND CONDITIONS START -->

        <!-- NOTE AND TERMS AND CONDITIONS END -->

        <!-- CANCEL SAVE SEND START -->
        <x-form-actions class="c-inv-btns d-block d-lg-flex d-md-flex">
            <div class="d-flex mb-3 mb-lg-0 mb-md-0">

                <div class="inv-action dropup mr-3">
                    <button class="btn-primary dropdown-toggle" type="button" data-toggle="dropdown"
                        aria-haspopup="true" aria-expanded="false">
                        @lang('app.save')
                        <span><i class="fa fa-chevron-up f-15 text-white"></i></span>
                    </button>
                    <!-- DROPDOWN - INFORMATION -->
                    <ul class="dropdown-menu" aria-labelledby="dropdownMenuBtn" tabindex="0">
                        <li>
                            <a class="dropdown-item f-14 text-dark save-form" href="javascript:;" data-type="save">
                                <i class="fa fa-save f-w-500 mr-2 f-11"></i> @lang('app.save')
                            </a>
                        </li>
                        <li>
                            <a class="dropdown-item f-14 text-dark save-form" href="javascript:void(0);"
                                data-type="send">
                                <i class="fa fa-paper-plane f-w-500  mr-2 f-12"></i> @lang('app.saveSend')
                            </a>
                        </li>
                    </ul>
                </div>

                <x-forms.button-secondary data-type="draft" class="save-form mr-3">@lang('app.saveDraft')
                </x-forms.button-secondary>

            </div>

            <x-forms.button-cancel :link="route('inspection_schedules.index')" class="border-0 ">@lang('app.cancel')
            </x-forms.button-cancel>

        </x-form-actions>
        <!-- CANCEL SAVE SEND END -->

    </x-form>
    <!-- FORM END -->
</div>
<!-- CREATE INVOICE END -->
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
            ...datepickerConfig
        });

        $('#awal, #akhir').timepicker({
            @if (company()->time_format == 'H:i')
                showMeridian: false,
            @endif
        }).on('hide.timepicker', function(e) {
            calculateTime();
        });

        $(document).on('click', '#add-item', function() {

            var i = $(document).find('.item_name').length;
            var item =
                `<div class="d-flex px-4 py-3 c-inv-desc item-row">
                <div class="c-inv-desc-table w-100 d-lg-flex d-md-flex d-block">
                <input type="text" class="form-control f-14 border-0 w-100 item_name" name="item_name[]" placeholder="@lang("modules.expenses.itemName")">
                </div>
                <a href="javascript:;" class="d-flex align-items-center justify-content-center ml-3 remove-item"><i class="fa fa-times-circle f-20 text-lightest"></i></a>
                </div>`;
            $(item).hide().appendTo("#sortable").fadeIn(500);
            $('#multiselect' + i).selectpicker();



        });

        $('#saveScheduleForm').on('click', '.remove-item', function() {
            $(this).closest('.item-row').fadeOut(300, function() {
                $(this).remove();
                $('select.customSequence').each(function(index) {
                    $(this).attr('name', 'taxes[' + index + '][]');
                    $(this).attr('id', 'multiselect' + index + '');
                });
                calculateTotal();
            });
        });

        $('.save-form').click(function() {
            var type = $(this).data('type');

            if (KTUtil.isMobileDevice()) {
                $('.desktop-description').remove();
            } else {
                $('.mobile-description').remove();
            }




            $.easyAjax({
                url: "{{ route('inspection_schedules.store') }}" + "?type=" + type,
                container: '#saveScheduleForm',
                type: "POST",
                blockUI: true,
                redirect: true,
                file: true,  // Commented so that we dot get error of Input variables exceeded 1000
                data: $('#saveScheduleForm').serialize(),
                success: function(response) {
                    if (response.status === 'success') {
                        window.location.href = response.redirectUrl;
                    }
                }
            })
        });

        $('#saveScheduleForm').on('click', '.remove-item', function() {
            $(this).closest('.item-row').fadeOut(300, function() {
                $(this).remove();
                $('select.customSequence').each(function(index) {
                    $(this).attr('name', 'taxes[' + index + '][]');
                    $(this).attr('id', 'multiselect' + index + '');
                });
                calculateTotal();
            });
        });






        init(RIGHT_MODAL);

        if (defaultClient != "") {
            changeClient(defaultClient);
        }
    });

    function ucWord(str){
            str = str.toLowerCase().replace(/\b[a-z]/g, function(letter) {
                return letter.toUpperCase();
            });
            return str;
        }

    function checkboxChange(parentClass, id) {
        var checkedData = '';
        $('.' + parentClass).find("input[type= 'checkbox']:checked").each(function() {
            checkedData = (checkedData !== '') ? checkedData + ', ' + $(this).val() : $(this).val();
        });
        $('#' + id).val(checkedData);
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
