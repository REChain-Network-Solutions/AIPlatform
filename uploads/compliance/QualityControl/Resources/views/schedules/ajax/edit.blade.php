@extends('layouts.app')

@section('content')
    <div class="content-wrapper">
        <!-- CREATE INVOICE START -->
        <div class="bg-white rounded b-shadow-4 create-inv">
            <!-- HEADING START -->
            <div class="px-lg-4 px-md-4 px-3 py-3">
                <h4 class="mb-0 f-21 font-weight-normal text-capitalize">Schedule @lang('app.details')</h4>
            </div>
            <!-- HEADING END -->
            <hr class="m-0 border-top-grey">
            <!-- FORM START -->
            <x-form class="c-inv-form" id="saveScheduleForm"> @method('PUT')

                <!-- INVOICE NUMBER, DATE, DUE DATE, FREQUENCY START -->
                <div class="row px-lg-4 px-md-4 px-3 py-3">
                    <!-- INVOICE NUMBER START -->
                    <div class="col-md-4">
                        <div class="form-group mb-lg-0 mb-md-0 mb-4">
                            <x-forms.text fieldId="subject" :fieldLabel="__('Schedule Job')"
                                :fieldValue="$schedule->subject"   fieldName="subject" fieldRequired="true" fieldPlaceholder="e.g. Toilet 1st floor etc.">
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
                                value="{{ $schedule->issue_date->translatedFormat(company()->date_format) }}">
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
                            <select class="form-control select-picker" name="floor_id" id="floor_id"
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

                </div>
                <!-- INVOICE NUMBER, DATE, DUE DATE, FREQUENCY END -->

                <hr class="m-0 border-top-grey">

                <!-- CLIENT, PROJECT, GST, BILLING, SHIPPING ADDRESS START -->
                <div class="row px-lg-4 px-md-4 px-3 pt-3">
                    <div class="col-md-4">
                        <x-forms.text fieldId="lokasi" :fieldLabel="__('Location')"
                            :fieldValue="$schedule->lokasi" fieldName="lokasi" fieldRequired="true" fieldPlaceholder="">
                        </x-forms.text>
                    </div>
                    <div class="col-md-1">
                        <x-forms.text fieldId="shift" :fieldLabel="__('Shift')"
                            :fieldValue="$schedule->shift" fieldName="shift" fieldRequired="true" fieldPlaceholder="">
                        </x-forms.text>
                    </div>

                    <div class="col-md-2">
                        <div class="bootstrap-timepicker timepicker">
                            <x-forms.text :fieldLabel="__('Start Time')"
                                :fieldPlaceholder="__('placeholders.hours')" fieldName="awal"
                                fieldId="awal" fieldRequired="true" :fieldValue="$schedule->awal" />
                        </div>
                    </div>

                    <div class="col-md-2">
                        <div class="bootstrap-timepicker timepicker">
                            <x-forms.text :fieldLabel="__('Finish Time')"
                                :fieldPlaceholder="__('placeholders.hours')" fieldName="akhir"
                                fieldId="akhir" fieldRequired="true" :fieldValue="$schedule->akhir" />
                        </div>
                    </div>
                    <div class="col-md-2 col-lg-3">
                        <x-forms.select fieldId="worker_id" :fieldLabel="'Choose Worker'" fieldName="worker_id" search="true"
                            fieldRequired="true">
                            <option value="">--</option>
                            @foreach ($employees as $employee)
                                <x-user-option :user="$employee"
                                                :selected="(request()->has('default_assign') && request('default_assign') == $employee->id) || ($schedule->worker_id == $employee->id)">
                                </x-user-option>
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
                                    <input type="hidden" name="item_ids[]" value="{{ $item->id }}">
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
                {{-- <div class="row px-lg-4 px-md-4 px-3 pb-3 pt-0 mb-3  mt-2">
                    <div class="col-md-12">
                        <a class="f-15 f-w-500" href="javascript:;" id="add-item"><i
                                class="icons icon-plus font-weight-bold mr-1"></i>@lang('modules.inspection_schedules.addItem')</a>
                    </div>
                </div> --}}
                <!--  ADD ITEM END-->

                <hr class="m-0 border-top-grey">

                <!-- TOTAL, DISCOUNT START -->

                <!-- TOTAL, DISCOUNT END -->

                <!-- NOTE AND TERMS AND CONDITIONS START -->

                <!-- NOTE AND TERMS AND CONDITIONS END -->

                <!-- CANCEL SAVE SEND START -->
                <div class="px-lg-4 px-md-4 px-3 py-3 c-inv-btns">
                    <div class="d-flex">
                        <x-forms.button-cancel :link="route('inspection_schedules.index')"
                                            class="border-0 mr-2">@lang('app.cancel')
                        </x-forms.button-cancel>
                        <x-forms.button-primary class="save-form"
                                                icon="check">@lang('app.save')
                        </x-forms.button-primary>
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
                $('#select' + i).selectpicker();
            });

            $('#saveScheduleForm').on('click', '.remove-item', function() {
                $(this).closest('.item-row').fadeOut(300, function() {
                    $(this).remove();

                });
            });

            $('.save-form').click(function() {
                $.easyAjax({
                    url: "{{ route('inspection_schedules.update', $schedule->id) }}",
                    container: '#saveScheduleForm',
                    type: "POST",
                    disableButton: true,
                    blockUI: true,
                    file: true,  // Commented so that we dot get error of Input variables exceeded 1000
                    data: $('#saveScheduleForm').serialize(),
                    success: function(response) {
                        if (response.status === 'success') {

                                window.location.href = response.redirectUrl;

                        }
                    }
                })
            });

            init(RIGHT_MODAL);
        });
    </script>
@endpush
