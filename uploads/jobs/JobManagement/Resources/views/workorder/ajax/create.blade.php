<link rel="stylesheet" href="{{ asset('vendor/css/dropzone.min.css') }}">
<!-- CREATE INVOICE START -->
<div class="bg-white rounded b-shadow-4 create-inv">
    <!-- HEADING START -->
    <div class="px-lg-4 px-md-4 px-3 py-3">
        <h4 class="mb-0 f-21 font-weight-normal text-capitalize">@lang('engineerings::modules.wo') @lang('app.details')</h4>
    </div>
    <!-- HEADING END -->
    <hr class="m-0 border-top-grey">
    <!-- FORM START -->
    <x-form class="c-inv-form" id="saveInvoiceForm">
        <div class="row px-lg-4 px-md-4 px-3 py-3">
            <div class="col-md-3">
                <div class="form-group mb-lg-0 mb-md-0 mb-4">
                    <x-forms.label class="mb-12" fieldId="nomor_wo" :fieldLabel="__('engineerings::app.menu.noWO')" fieldRequired="true">
                    </x-forms.label>
                    <x-forms.input-group>
                        <input type="text" name="nomor_wo" id="nomor_wo" class="form-control height-35 f-15"
                            value="{{ $nomor }}">
                    </x-forms.input-group>
                </div>
            </div>
            <div class="col-md-3">
                <div class="form-group mb-lg-0 mb-md-0 mb-4">
                    <x-forms.label fieldId="workrequest_id" :fieldLabel="__('engineerings::app.menu.WRid')" fieldName="workrequest_id"
                        fieldRequired="true">
                    </x-forms.label>
                    <x-forms.input-group>
                        <select class="form-control select-picker" name="workrequest_id" id="workrequest_id"
                            data-live-search="true">
                            <option value="">--</option>
                            @foreach ($wr as $items)
                                <option value="{{ $items->id }}">{{ ucwords($items->wr_no) }}</option>
                            @endforeach
                        </select>
                    </x-forms.input-group>
                </div>
            </div>
            <div class="col-md-3">
                <div class="form-group mb-lg-0 mb-md-0 mb-4">
                    <x-forms.label fieldId="complaint_id" :fieldLabel="__('engineerings::app.menu.ticketID')" fieldName="complaint_id"
                        fieldRequired="true">
                    </x-forms.label>
                    <x-forms.input-group>
                        <select class="form-control select-picker" name="complaint_id" id="complaint_id"
                            data-live-search="true">
                            <option value="">--</option>
                            @foreach ($ticket as $items)
                                <option value="{{ $items->id }}">{{ ucwords($items->subject) }}</option>
                            @endforeach
                        </select>
                    </x-forms.input-group>
                </div>
            </div>
            <div class="col-md-3">
                <div class="form-group mb-lg-0 mb-md-0 mb-4">
                    <x-forms.label fieldId="invoice_id" :fieldLabel="__('engineerings::app.menu.invoiceID')" fieldName="invoice_id" fieldRequired="true">
                    </x-forms.label>
                    <x-forms.input-group>
                        <select class="form-control select-picker" name="invoice_id" id="invoice_id"
                            data-live-search="true">
                            <option value="">--</option>
                            @foreach ($invoice as $items)
                                <option value="{{ $items->id }}">{{ ucwords($items->custom_invoice_number) }}
                                </option>
                            @endforeach
                        </select>
                    </x-forms.input-group>
                </div>
            </div>

            <div class="col-md-9">
                <div class="form-group mb-lg-0 mb-md-0 mb-4 mt-4">
                    <x-forms.label class="mb-12" fieldId="problem" :fieldLabel="__('engineerings::app.menu.problem')" fieldRequired="true">
                    </x-forms.label>
                    <x-forms.input-group>
                        <input type="text" name="problem" id="problem" class="form-control height-35 f-15"
                            value="">
                    </x-forms.input-group>
                </div>
            </div>
            <div class="col-md-6">
                <x-forms.label class="mt-3" fieldId="parent_label" :fieldLabel="__('complaint::app.menu.area')" fieldName="parent_label">
                </x-forms.label>
                <select class="form-control select-picker" name="area_id" id="area_id" data-live-search="true">
                    <option value="">No Items Selected</option>
                    @foreach ($areas as $area)
                        <option value="{{ $area->id }}">{{ $area->area_name }}</option>
                    @endforeach
                </select>
            </div>
            <div class="col-md-6">
                <x-forms.label class="mt-3" fieldId="parent_label" :fieldLabel="__('complaint::app.menu.houses')" fieldName="parent_label">
                </x-forms.label>
                <select class="form-control select-picker" name="house_id" id="house_id" data-row-id="0"
                    data-live-search="true">
                    <option value="">No Items Selected</option>
                </select>
                </select>
            </div>

            <div class="col-lg-3">
                <x-forms.label class="mt-3" fieldId="assets_id" :fieldLabel="__('engineerings::app.menu.assets')"
                    fieldRequired="true"></x-forms.label>
                <x-forms.input-group>
                    <select class="form-control select-picker" name="assets_id" id="assets_id">
                        <option value="">--</option>
                    </select>
                </x-forms.input-group>
            </div>
            <div class="col-lg-3">
                <x-forms.label class="mt-3" fieldId="category" :fieldLabel="__('engineerings::app.menu.category')"
                    fieldRequired="true"></x-forms.label>
                <x-forms.input-group>
                    <select class="form-control select-picker" name="category" id="category">
                        <option value="">--</option>
                        <option value="planned">Planned</option>
                        <option value="unplanned">Unplanned</option>
                    </select>
                </x-forms.input-group>
            </div>
            <div class="col-lg-3">
                <x-forms.label class="mt-3" fieldId="priority" :fieldLabel="__('engineerings::app.menu.priority')"
                    fieldRequired="true"></x-forms.label>
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
            <div class="col-lg-3">
                <x-forms.label class="mt-3" fieldId="status" :fieldLabel="__('engineerings::app.menu.status')"
                    fieldRequired="true"></x-forms.label>
                <x-forms.input-group>
                    <select class="form-control select-picker" name="status" id="status">
                        <option value="">--</option>
                        <option value="incomplete">Incompleted</option>
                        <option value="pending">Pending</option>
                        <option value="completed">Completed</option>
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

            <div class="col-lg-4">
                <x-forms.label class="mt-3" fieldId="actual_start" :fieldLabel="__('engineerings::app.menu.actualStart')" fieldRequired="true">
                </x-forms.label>
                <div class="bootstrap-timepicker timepicker">
                    <input type="datetime-local" id="actual_start" name="actual_start"
                        class="px-6 position-relative text-dark font-weight-normal form-control height-35 rounded p-0 text-left f-15">
                </div>
            </div>
            <div class="col-lg-4">
                <x-forms.label class="mt-3" fieldId="actual_finish" :fieldLabel="__('engineerings::app.menu.actualFinish')" fieldRequired="true">
                </x-forms.label>
                <div class="bootstrap-timepicker timepicker">
                    <input type="datetime-local" id="actual_finish" name="actual_finish"
                        class="px-6 position-relative text-dark font-weight-normal form-control height-35 rounded p-0 text-left f-15">
                </div>
            </div>
            <div class="col-md-4">
                <div class="form-group mb-lg-0 mb-md-0 mb-4">
                    <x-forms.label class="mt-3" fieldId="actual_hours" :fieldLabel="__('engineerings::app.menu.actualHours')"></x-forms.label>
                    <x-forms.input-group class="border px-2">
                        <label for="">@lang('engineerings::app.menu.hours'): </label>
                        <input type="text" name="actual_hours" id="actual_hours"
                            class="form-control-plaintext height-35 f-15 px-2 border-right" value="" readonly>
                        <label class="px-2">@lang('engineerings::app.menu.min'): </label>
                        <input type="text" name="actual_minutes" id="actual_minutes"
                            class="form-control-plaintext height-35 f-15 px-2" value="" readonly>
                    </x-forms.input-group>
                </div>
            </div>

            <div class="col-md-6">
                <div class="form-group">
                    <x-forms.label class="my-3" fieldId="description-text" :fieldLabel="__('engineerings::app.menu.workDesc')" fieldRequired="true">
                    </x-forms.label>
                    <textarea name="work_description" id="description-text" rows="5" class="form-control"></textarea>
                </div>
            </div>
            <div class="col-md-6">
                <div class="form-group">
                    <x-forms.label class="my-3" fieldId="description-text" :fieldLabel="__('engineerings::app.menu.completitionNotes')" fieldRequired="true">
                    </x-forms.label>
                    <textarea name="completion_notes" id="description-text" rows="5" class="form-control"></textarea>
                </div>
            </div>
            <div class="col-lg-12">
                <x-forms.file-multiple class="mr-0 mr-lg-2 mr-md-2" :fieldLabel="__('app.add') . ' ' . __('app.file')" fieldName="file"
                    fieldId="file-upload-dropzone" />
                <input type="hidden" name="workorderID" id="workorderID">
            </div>
        </div>

        <!-- CANCEL SAVE START -->
        <x-form-actions class="c-inv-btns d-block d-lg-flex d-md-flex">
            <x-forms.button-primary data-type="save" class="save-form mr-3" icon="check">@lang('app.save')
            </x-forms.button-primary>

            <x-forms.button-cancel :link="route('work.index')" class="border-0 ">@lang('app.cancel')
            </x-forms.button-cancel>
        </x-form-actions>
        <!-- CANCEL SAVE END -->

    </x-form>
    <!-- FORM END -->
</div>
<script src="{{ asset('vendor/jquery/dropzone.min.js') }}"></script>
<script>
    $(document).ready(function() {
        $(document).on('change', 'select[name="area_id"]', function() {
            var selectedId = $(this).val();
            houseItems(selectedId);
        });

        function houseItems(selectedId) {
            $('.select-picker').selectpicker();
            $.ajax({
                url: '{{ route('complaint.get-items', '') }}' + '/' + selectedId,
                method: 'GET',
                success: function(data) {
                    $('select[name^="house_id"]').each(function(index, select) {
                        var $select = $(select);
                        var selectedValue = $select.val();
                        var parentRow = $(this).closest('tr');
                        var qtyInput = parentRow.find('input[name="house_id"]');
                        $select.empty();
                        $.each(data, function(idx, item) {
                            var option = $('<option>', {
                                value: item.id,
                                text: item.house_name
                            });
                            $select.append(option);
                        });
                        $select.val(selectedValue);
                        $select.selectpicker('refresh');
                        qtyInput.val('');
                    });
                }
            });
        }

        $('#house_id').change(function(e) {
            getAssets()
        });

        function getAssets() {
            var house_id = document.getElementById("house_id").value;
            var url = "{{ route('work.get_assets', ':id') }}";
            url = (house_id) ? url.replace(':id', house_id) : url.replace(':id', null);
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
        var actstartInput = document.getElementById("actual_start");
        var actfinishInput = document.getElementById("actual_finish");

        // Ambil elemen input untuk estimate_hours dan estimate_minutes
        var jamInput = document.getElementById("estimate_hours");
        var minInput = document.getElementById("estimate_minutes");
        var actjamInput = document.getElementById("actual_hours");
        var actminInput = document.getElementById("actual_minutes");

        // Tambahkan event listener pada kedua elemen input
        startInput.addEventListener("input", updateEstimate);
        finishInput.addEventListener("input", updateEstimate);
        actstartInput.addEventListener("input", updateActual);
        actfinishInput.addEventListener("input", updateActual);

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

        function updateActual() {
            var actstart = new Date(actstartInput.value);
            var actfinish = new Date(actfinishInput.value);
            var actdiff = actfinish - actstart;
            var acthours = actdiff / (1000 * 60 * 60);
            var actjam = Math.floor(acthours);
            var actmin = Math.floor((acthours % 1) * 60);

            // Tampilkan hasil pada input estimate_hours dan estimate_minutes
            actjamInput.value = actjam;
            actminInput.value = actmin;
        }

        Dropzone.autoDiscover = false;
        //Dropzone class
        myDropzone = new Dropzone("div#file-upload-dropzone", {
            dictDefaultMessage: "{{ __('app.dragDrop') }}",
            url: "{{ route('work.multiple_upload') }}",
            headers: {
                'X-CSRF-TOKEN': '{{ csrf_token() }}'
            },
            paramName: "file",
            maxFilesize: DROPZONE_MAX_FILESIZE,
            maxFiles: 10,
            autoProcessQueue: false,
            uploadMultiple: true,
            addRemoveLinks: true,
            parallelUploads: 10,
            acceptedFiles: DROPZONE_FILE_ALLOW,
            init: function() {
                myDropzone = this;
            }
        });
        myDropzone.on('sending', function(file, xhr, formData) {
            var ids = $('#workorderID').val();
            formData.append('workorderID', ids);
        });
        myDropzone.on('uploadprogress', function() {
            $.easyBlockUI();
        });
        myDropzone.on('completemultiple', function() {
            var msgs = "@lang('messages.updateSuccess')";
            window.location.href = "{{ route('work.index') }}"
        });

        $('.save-form').click(function() {
            $.easyAjax({
                url: "{{ route('work.store') }}",
                container: '#saveInvoiceForm',
                type: "POST",
                blockUI: true,
                redirect: true,
                file: true,
                data: $('#saveInvoiceForm').serialize(),
                success: function(response) {
                    if (response.status == 'success') {
                        if (myDropzone.getQueuedFiles().length > 0) {
                            $('#workorderID').val(response.workorderID);
                            myDropzone.processQueue();
                        } else {
                            window.location.href =
                                "{{ route('work.index') }}";
                        }
                    }
                }
            });
        });

        init(RIGHT_MODAL);
    });
</script>
