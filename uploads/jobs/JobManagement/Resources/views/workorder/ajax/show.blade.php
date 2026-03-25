<link rel="stylesheet" href="{{ asset('vendor/css/dropzone.min.css') }}">
<div class="row mt-4">
    <div class="col-xl-7 col-lg-12 col-md-12 mb-4 mb-xl-0 mb-lg-4">
        <x-cards.data>
            <div class="row mb-3">
                <div class="col-10">
                    <h4 class="card-title f-15 f-w-500 text-darkest-grey mb-0">
                        @lang('engineerings::modules.wo') @lang('app.details')
                    </h4>
                </div>
                <div class="col-2 text-right">
                    <div class="dropdown">
                        <button class="btn f-14 px-0 py-0 text-dark-grey dropdown-toggle" type="button"
                            data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">
                            <i class="fa fa-ellipsis-h"></i>
                        </button>

                        <div class="dropdown-menu dropdown-menu-right border-grey rounded b-shadow-4 p-0"
                            aria-labelledby="dropdownMenuLink" tabindex="0">
                            <a class="dropdown-item" href="{{ route('work.edit', $wo->id) }}">@lang('app.edit')</a>
                            <a class="dropdown-item delete-unit">@lang('app.delete')</a>
                        </div>
                    </div>
                </div>
            </div>
            <x-cards.data-row :label="__('engineerings::app.menu.noWO')" :value="$wo->nomor_wo" />
            @if ($wo->workrequest_id != '')
                <x-cards.data-row :label="__('engineerings::app.menu.WRid')" :value="$wo->wr->wr_no" />
            @endif
            @if ($wo->complaint_id != '')
                <x-cards.data-row :label="__('engineerings::app.menu.ticketID')" :value="mb_ucwords($wo->ticket->subject)" />
            @endif
            @if ($wo->invoice_id != '')
                <x-cards.data-row :label="__('engineerings::app.menu.invoiceID')" :value="mb_ucwords($wo->invoice->custom_invoice_number)" />
            @endif
            <x-cards.data-row :label="__('complaint::app.menu.area')" :value="mb_ucwords(optional(optional($wo->house)->area)->area_name ?? '--')" />
            <x-cards.data-row :label="__('complaint::app.menu.houses')" :value="mb_ucwords(optional($wo->house)->house_name ?? '--')" />
            <x-cards.data-row :label="__('engineerings::app.menu.assets')" :value="mb_ucwords(optional(optional($wo->assets)->type)->type_name ?? '--')" />
            <x-cards.data-row :label="__('engineerings::app.menu.category')" :value="mb_ucwords($wo->category ?? '--')" />
            <x-cards.data-row :label="__('engineerings::app.menu.priority')" :value="mb_ucwords($wo->priority ?? '--')" />
            <x-cards.data-row :label="__('engineerings::app.menu.status')" :value="mb_ucwords($wo->status ?? '--')" />
            <x-cards.data-row :label="__('engineerings::app.menu.scheduleStart')" :value="mb_ucwords($wo->schedule_start ?? '--')" />
            <x-cards.data-row :label="__('engineerings::app.menu.scheduleFinish')" :value="mb_ucwords($wo->schedule_finish ?? '--')" />
            <x-cards.data-row :label="__('engineerings::app.menu.estimateHours')" :value="mb_ucwords(($wo->estimate_hours ?? '--') . ' hours,' . ($wo->estimate_minutes ?? '--'))" />
            <x-cards.data-row :label="__('engineerings::app.menu.actualStart')" :value="$wo->actual_start ?? '--'" />
            <x-cards.data-row :label="__('engineerings::app.menu.actualFinish')" :value="$wo->actual_finish ?? '--'" />
            <x-cards.data-row :label="__('engineerings::app.menu.actualHours')" :value="($wo->actual_hours ?? '--') . ' hours,' . ($wo->actual_minutes ?? '--')" />
            <x-cards.data-row :label="__('engineerings::app.menu.workDesc')" :value="mb_ucwords($wo->work_description ?? '--')" />
            <x-cards.data-row :label="__('engineerings::app.menu.completitionNotes')" :value="mb_ucwords($wo->completion_notes ?? '--')" />

        </x-cards.data>
    </div>
    <div class="col-xl-5 col-lg-12 col-md-12 ">
        <x-cards.data>
            <div class="row mb-3">
                <div class="col-10">
                    <h4 class="card-title f-15 f-w-500 text-darkest-grey mb-0">
                        @lang('engineerings::modules.wo') @lang('Files')
                    </h4>
                </div>
            </div>
            <div class="row" id="add-btn">
                <div class="col-md-12">
                    <a class="f-15 f-w-500" href="javascript:;" id="add-task-file"><i
                            class="icons icon-plus font-weight-bold mr-1"></i>@lang('modules.projects.uploadFile')</a>
                </div>
            </div>
            <x-form id="save-taskfile-data-form" class="d-none">
                <input type="hidden" name="project_id" value="{{ $wo->id }}">
                <div class="row">
                    <div class="col-md-12">
                        <x-forms.file-multiple :fieldLabel="__('modules.projects.uploadFile')" fieldName="file" fieldId="employee_file" />
                    </div>
                    <div class="col-md-12">
                        <div class="w-100 justify-content-end d-flex mt-2">
                            <x-forms.button-cancel id="cancel-taskfile" class="border-0">@lang('app.cancel')
                            </x-forms.button-cancel>
                        </div>
                    </div>
                </div>
            </x-form>

            <div class="d-flex flex-wrap mt-3" id="task-file-list">
                @forelse($wo->files as $file)
                    <x-file-card :fileName="$file->filename" :dateAdded="$file->created_at->diffForHumans()">
                        @if ($file->icon == 'images')
                            <img src="{{ $file->file_url }}">
                        @else
                            <i class="fa {{ $file->icon }} text-lightest"></i>
                        @endif

                        <x-slot name="action">
                            <div class="dropdown ml-auto file-action">
                                <button
                                    class="btn btn-lg f-14 p-0 text-lightest text-capitalize rounded  dropdown-toggle"
                                    type="button" data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">
                                    <i class="fa fa-ellipsis-h"></i>
                                </button>

                                <div class="dropdown-menu dropdown-menu-right border-grey rounded b-shadow-4 p-0"
                                    aria-labelledby="dropdownMenuLink" tabindex="0">
                                    <a class="cursor-pointer d-block text-dark-grey f-13 pt-3 px-3 " target="_blank"
                                        href="{{ $file->file_url }}">@lang('app.view')</a>

                                    <a class="cursor-pointer d-block text-dark-grey f-13 py-3 px-3 "
                                        href="{{ route('files.download', md5($file->id)) }}">@lang('app.download')</a>
                                    <a class="cursor-pointer d-block text-dark-grey f-13 pb-3 px-3 delete-file"
                                        data-row-id="{{ $file->id }}" href="javascript:;">@lang('app.delete')</a>
                                </div>
                            </div>
                        </x-slot>
                    </x-file-card>
                @empty
                    <div class="align-items-center d-flex flex-column text-lightest p-20 w-100">
                        <i class="fa fa-file-excel f-21 w-100"></i>
                        <div class="f-15 mt-4">
                            @lang('messages.noFileUploaded')
                        </div>
                    </div>
                @endforelse
            </div>
        </x-cards.data>
    </div>
</div>
<!-- ROW END -->

<script src="{{ asset('vendor/jquery/dropzone.min.js') }}"></script>
<script>
    $(document).ready(function() {
        Dropzone.autoDiscover = false;
        taskDropzone = new Dropzone("#employee_file", {
            dictDefaultMessage: "{{ __('app.dragDrop') }}",
            url: "{{ route('work-file.store') }}",
            headers: {
                'X-CSRF-TOKEN': '{{ csrf_token() }}'
            },
            paramName: "file",
            maxFilesize: DROPZONE_MAX_FILESIZE,
            maxFiles: 10,
            timeout: 0,
            uploadMultiple: true,
            addRemoveLinks: true,
            parallelUploads: 10,
            acceptedFiles: DROPZONE_FILE_ALLOW,
            init: function() {
                taskDropzone = this;
            }
        });
        taskDropzone.on('sending', function(file, xhr, formData) {
            var ids = "{{ $wo->id }}";
            formData.append('workorderID', ids);
            $.easyBlockUI();
        });
        taskDropzone.on('uploadprogress', function() {
            $.easyBlockUI();
        });
        taskDropzone.on('completemultiple', function(file) {
            var taskView = JSON.parse(file[0].xhr.response).view;
            taskDropzone.removeAllFiles();
            $.easyUnblockUI();
            $('#task-file-list').html(taskView);
        });
        taskDropzone.on('error', function(file) {});

        $('#add-task-file').click(function() {
            $(this).closest('.row').addClass('d-none');
            $('#save-taskfile-data-form').removeClass('d-none');
        });

        $('#cancel-document').click(function() {
            $('#save-taskfile-data-form').addClass('d-none');
            $('#add-task-file').closest('.row').removeClass('d-none');
        });

        $('body').on('click', '#cancel-taskfile', function() {
            $('#save-taskfile-data-form').toggleClass('d-none');
            $('#add-btn').toggleClass('d-none');
        });


        $('body').on('click', '.delete-file', function() {
            var id = $(this).data('row-id');
            Swal.fire({
                title: "@lang('messages.sweetAlertTitle')",
                text: "@lang('messages.recoverRecord')",
                icon: 'warning',
                showCancelButton: true,
                focusConfirm: false,
                confirmButtonText: "@lang('messages.confirmDelete')",
                cancelButtonText: "@lang('app.cancel')",
                customClass: {
                    confirmButton: 'btn btn-primary mr-3',
                    cancelButton: 'btn btn-secondary'
                },
                showClass: {
                    popup: 'swal2-noanimation',
                    backdrop: 'swal2-noanimation'
                },
                buttonsStyling: false
            }).then((result) => {
                if (result.isConfirmed) {
                    var url = "{{ route('work-file.destroy', ':id') }}";
                    url = url.replace(':id', id);

                    var token = "{{ csrf_token() }}";

                    $.easyAjax({
                        type: 'POST',
                        url: url,
                        data: {
                            '_token': token,
                            '_method': 'DELETE'
                        },
                        success: function(response) {
                            if (response.status == "success") {
                                $('#task-file-list').html(response.view);
                            }
                        }
                    });
                }
            });
        });

    });
</script>
