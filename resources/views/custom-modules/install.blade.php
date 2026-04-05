@extends('layouts.app')

@push('styles')
    <link rel="stylesheet" href="{{ asset('vendor/css/dropzone.min.css') }}">
    <link rel="stylesheet" href="{{ asset('public/css/custom-modules/pass58.css') }}">
    <style>
        .dropzone .dz-preview .dz-error-message {
            top: 150px !important;
        }

        #install-process .alert-danger,
        .module-diagnostic-report .alert-danger,
        .module-diagnostic-report .badge-danger {
            background: #5c1118 !important;
            border-color: #7b1d26 !important;
            color: #fff !important;
        }

        #install-process .alert-warning,
        .module-diagnostic-report .alert-warning,
        .module-diagnostic-report .badge-warning {
            background: #5a4206 !important;
            border-color: #7a5a10 !important;
            color: #fff3cd !important;
        }

        #install-process .alert-info,
        .module-diagnostic-report .alert-info {
            background: #123a55 !important;
            border-color: #1c587f !important;
            color: #d9f1ff !important;
        }

        .module-diagnostic-report .alert-light,
        #install-process .alert-light {
            background: #111a33 !important;
            border-color: #263357 !important;
            color: #dfe8ff !important;
        }

        .module-diagnostic-report .table {
            color: #eef2ff;
        }

        .module-diagnostic-report .table thead th,
        .module-diagnostic-report .table td,
        .module-diagnostic-report .table th {
            border-color: #33406b !important;
        }


        .s-b-n-header,
        .s-b-n-header .border-bottom-grey {
            background: #0f172a !important;
            color: #ffffff !important;
            border-color: #24324f !important;
        }

        .s-b-n-header h2,
        .s-b-n-header h4,
        .s-b-n-header .f-21,
        .s-b-n-header .font-weight-normal {
            color: #ffffff !important;
            white-space: normal !important;
            word-break: break-word;
        }

        .alert.alert-info {
            background: #123a55 !important;
            border-color: #1c587f !important;
            color: #d9f1ff !important;
        }
    </style>
@endpush

@section('content')

    <!-- SETTINGS START -->
    <div class="w-100 d-flex ">

        @include('sections.setting-sidebar')

        <x-setting-card>
            <x-slot name="header">
                <div class="s-b-n-header" id="tabs">
                    <h2 class="mb-0 p-20 f-21 font-weight-normal  border-bottom-grey">
                        {{ __($pageTitle) === $pageTitle ? 'Custom Module Installer' : __($pageTitle) }}</h2>
                </div>
            </x-slot>

            <div class="col-lg-12 col-md-12 ntfcn-tab-content-left w-100 p-4 ">
                <h4 class="f-21 font-weight-normal  ">
                    @lang('modules.moduleSettings.step1')</h4>


                <div class="row">
                    <div class="col-sm-12">

                        @php
                            $uploadMaxFilesize = \App\Helper\Files::getUploadMaxFilesize();
                            $postMaxSize = \App\Helper\Files::getPostMaxSize();
                        @endphp

                        @if(!$uploadMaxFilesize['greater'])
                            <span class="text-danger">
                                    Your Server upload_max_filesize = {{\App\Helper\Files::getUploadMaxFilesize()['size']}}.
                                    Please change to min <strong>{{\App\Helper\Files::REQUIRED_FILE_UPLOAD_SIZE}}MB</strong>
                                    to upload big modules
                            </span>
                        @elseif(!$postMaxSize['greater'])
                            <span class="text-danger">
                                    Your Server post_max_size = {{\App\Helper\Files::getPostMaxSize()['size']}}.
                                    Please change to min <strong>{{\App\Helper\Files::REQUIRED_FILE_UPLOAD_SIZE}}MB</strong> to
                                    upload big modules
                            </span>
                        @endif

                        <x-forms.file-multiple
                            class="mr-0 mr-lg-2 mr-md-2"
                            :fieldLabel=" __('messages.downloadFilefromCodecanyon') " fieldName="file"
                            fieldId="file-upload-dropzone"/>
                    </div>
                </div>
            </div>

            <div class="col-md-12 " id="install-process"></div>

            @include('custom-modules.partials.summary-cards')


            <div class="col-md-12 mb-4">
                <div class="card border">
                    <div class="card-body">
                        <h5 class="mb-3">Install options</h5>
                        <div class="row">
                            <div class="col-md-4">
                                <x-forms.label fieldId="package_attach_mode" :fieldLabel="'Package attachment'"/>
                                <select class="form-control" id="package_attach_mode" name="package_attach_mode">
                                    <option value="none">Do not change packages</option>
                                    <option value="all">Attach to all packages</option>
                                    <option value="selected">Attach to selected packages</option>
                                </select>
                            </div>
                            <div class="col-md-8" id="package_select_wrapper" style="display:none;">
                                <x-forms.label fieldId="package_ids" :fieldLabel="'Selected packages'"/>
                                <select class="form-control" id="package_ids" name="package_ids" multiple>
                                    @forelse(($packageChoices ?? collect()) as $package)
                                        <option value="{{ $package->id }}">{{ $package->package_label ?: ('Package #' . $package->id) }}</option>
                                    @empty
                                        <option value="" disabled>No packages available</option>
                                    @endforelse
                                </select>
                                <small class="text-muted">Used only when "Attach to selected packages" is chosen.</small>
                            </div>
                        </div>
                        <div class="row mt-3">
                            <div class="col-md-4">
                                <div class="custom-control custom-switch">
                                    <input type="checkbox" class="custom-control-input" id="seed_company_module_settings">
                                    <label class="custom-control-label" for="seed_company_module_settings">Seed company module settings</label>
                                </div>
                            </div>
                            <div class="col-md-4">
                                <div class="custom-control custom-switch">
                                    <input type="checkbox" class="custom-control-input" id="run_permission_seeders">
                                    <label class="custom-control-label" for="run_permission_seeders">Run permission seeders</label>
                                </div>
                            </div>
                            <div class="col-md-4">
                                <div class="custom-control custom-switch">
                                    <input type="checkbox" class="custom-control-input" id="enable_module_after_install" checked>
                                    <label class="custom-control-label" for="enable_module_after_install">Enable module after install</label>
                                </div>
                            </div>
                        </div>
                        <div class="row mt-3">
                            <div class="col-md-4">
                                <div class="custom-control custom-switch">
                                    <input type="checkbox" class="custom-control-input" id="run_module_migrations" checked>
                                    <label class="custom-control-label" for="run_module_migrations">Run module migrations</label>
                                </div>
                            </div>
                            <div class="col-md-4">
                                <div class="custom-control custom-switch">
                                    <input type="checkbox" class="custom-control-input" id="run_post_install_audit" checked>
                                    <label class="custom-control-label" for="run_post_install_audit">Run post-install doctor audit</label>
                                </div>
                            </div>
                            <div class="col-md-4">
                                <div class="custom-control custom-switch">
                                    <input type="checkbox" class="custom-control-input" id="auto_repair_metadata" checked>
                                    <label class="custom-control-label" for="auto_repair_metadata">Auto-repair missing Worksuite metadata</label>
                                    <small class="d-block text-muted mt-1">Includes package bridge, tenant module settings, and visibility-related metadata where safe.</small>
                                </div>
                            </div>
                            <div class="col-md-4 mt-3 mt-md-0">
                                <div class="custom-control custom-switch">
                                    <input type="checkbox" class="custom-control-input" id="run_safe_autofix" checked>
                                    <label class="custom-control-label" for="run_safe_autofix">Run safe autofix after install</label>
                                    <small class="d-block text-muted mt-1">Applies safe database repairs for permissions, package bridges, module settings, and tenant visibility.</small>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-lg-12 col-md-12 ntfcn-tab-content-left w-100 p-4 ">
                <h4 class="f-21 font-weight-normal ">
                    @lang('modules.moduleSettings.step2')</h4>
                <div class="alert alert-light border mt-3 mb-0">
                    <strong>User menu visibility checks are now enabled. Route truth, sidebar inclusion, permission DB readiness, role-pivot truth, extension/module registry truth, and tenant package entitlement checks now run in the analyzer.</strong> The installer will warn if the module can install but still stay hidden from tenant users due to sidebar, route, permission, role-mapping, extension-registry, module_settings, company_module_settings, or package bridge problems.
                </div>

                <p>@lang('modules.update.moduleFile')</p>
            </div>
            @include('custom-modules.partials.upload-summary-grid')

            <div class="col-md-12 mb-3">
                <ul class="list-group" id="files-list">
                    @php($moduleFilesForList = ($moduleFiles ?? \Illuminate\Support\Facades\File::files($updateFilePath)))
                    @forelse ($moduleFilesForList as $key => $filename)
                        @if (\Illuminate\Support\Facades\File::basename($filename) != 'modules_statuses.json' && strpos(\Illuminate\Support\Facades\File::basename($filename), 'auto') === false)
                            <li class="list-group-item" id="file-{{ $key + 1 }}">
                                <div class="row">
                                    <div class="col-lg-6 py-1">
                                        <b>{{ \Illuminate\Support\Facades\File::basename($filename) }}</b>
                                    </div>

                                    <div class="col-lg-4 py-1 text-center f-12">
                                        @lang('app.uploadDate'):
                                        {{ \Carbon\Carbon::parse(\Illuminate\Support\Facades\File::lastModified($filename))->timezone(global_setting()->timezone)->translatedFormat('jS M, Y g:i A') }}
                                    </div>

                                    <div class="col-lg-2 text-lg-right py-1">
                                        <button type="button"
                                                class="btn btn-secondary p-1 f-13 btn-sm mr-2 analyze-files"
                                                data-file-no="{{ $key + 1 }}"
                                                data-file-path="{{ $filename }}">Analyze ZIP <i class="fa fa-search"></i>
                                        </button>

                                        <button type="button"
                                                class="btn btn-outline-info p-1 f-13 btn-sm mr-2 export-analysis"
                                                data-file-no="{{ $key + 1 }}"
                                                data-file-path="{{ $filename }}">Export JSON <i class="fa fa-file-download"></i>
                                        </button>

                                        <button type="button"
                                                class="btn btn-outline-warning p-1 f-13 btn-sm mr-2 autofix-files"
                                                data-file-no="{{ $key + 1 }}"
                                                data-file-path="{{ $filename }}">Analyze + Safe Fix <i class="fa fa-wrench"></i>
                                        </button>

                                        <button type="button"
                                                class="btn btn-primary p-1 f-13 btn-sm mr-2 install-files"
                                                data-file-no="{{ $key + 1 }}"
                                                data-file-path="{{ $filename }}">Install if Ready <i
                                                class="fa fa-download"></i>
                                        </button>

                                        <button type="button"
                                                class="btn btn-light f-13 btn-sm delete-files"
                                                data-file-no="{{ $key + 1 }}" data-toggle="tooltip"
                                                data-original-title="@lang('app.delete')"
                                                data-file-path="{{ $filename }}">
                                            <i class="fa fa-trash"></i>
                                        </button>
                                    </div>
                                </div>
                            </li>
                        @endif
                    @endforeach
                </ul>
            </div>

            <x-slot name="action">
                <!-- Buttons Start -->
                <div class="w-100 border-top-grey">
                    <x-setting-form-actions>
                        <a href="{{ route('custom-modules.index').'?tab=custom' }}" class="btn-secondary rounded f-14 p-2">
                            @lang('app.back')
                        </a>
                    </x-setting-form-actions>
                    <div class="d-block d-lg-none d-md-none p-4">
                        <x-forms.button-cancel :link="route('custom-modules.index').'?tab=custom'" class="w-100 mt-3">
                            @lang('app.cancel')
                        </x-forms.button-cancel>
                    </div>
                </div>
                <!-- Buttons End -->
            </x-slot>

        </x-setting-card>

    </div>
    <!-- SETTINGS END -->

@endsection

@push('scripts')
    <script>
        Dropzone.autoDiscover = false;
        $(document).ready(function () {
            const uploadFile = "{{ route('update-settings.store') }}?_token={{ csrf_token() }}";

            const togglePackageSelect = function () {
                if ($('#package_attach_mode').val() === 'selected') {
                    $('#package_select_wrapper').show();
                }
                else {
                    $('#package_select_wrapper').hide();
                    $('#package_ids').val([]);
                }
            };
            const myDrop = new Dropzone("#file-upload-dropzone", {
                url: uploadFile,
                acceptedFiles: 'application/zip, application/x-zip-compressed, application/x-compressed, multipart/x-zip',
                addRemoveLinks: true,
                dictDefaultMessage: "@lang('app.dropFileToUpload')",
            });
            $('#package_attach_mode').on('change', togglePackageSelect);
            togglePackageSelect();

            myDrop.on("complete", function (file) {
                if (myDrop.getRejectedFiles().length == 0) {
                    window.location.reload();
                }
            });
        });

        $('.analyze-files').click(function () {

            $('#install-process').html('<div class="alert alert-primary">Analyzing ZIP package...</div>');

            let filePath = $(this).data('file-path');
            $.easyAjax({
                type: 'POST',
                url: "{{ route('custom-modules.analyze') }}",
                blockUI: true,
                data: {
                    "_token": "{{ csrf_token() }}",
                    filePath: filePath,
                    package_attach_mode: $('#package_attach_mode').val(),
                    package_ids: $('#package_ids').val(),
                    seed_company_module_settings: $('#seed_company_module_settings').is(':checked') ? 1 : 0,
                    run_permission_seeders: $('#run_permission_seeders').is(':checked') ? 1 : 0,
                    enable_module_after_install: $('#enable_module_after_install').is(':checked') ? 1 : 0,
                    run_module_migrations: $('#run_module_migrations').is(':checked') ? 1 : 0,
                    run_post_install_audit: $('#run_post_install_audit').is(':checked') ? 1 : 0,
                    auto_repair_metadata: $('#auto_repair_metadata').is(':checked') ? 1 : 0,
                    run_safe_autofix: $('#run_safe_autofix').is(':checked') ? 1 : 0
                },
                success: function (response) {
                    if (response.diagnostics) {
                        $('#install-process').html(response.diagnostics);
                    }
                    else if (response.message) {
                        $('#install-process').html(`<div class="alert alert-${response.status === 'success' ? 'success' : 'danger'}">${response.message}</div>`);
                    }
                    else {
                        $('#install-process').html('<div class="alert alert-danger">Analyzer did not return any diagnostics.</div>');
                    }
                }
            });
        });

        $('.install-files').click(function () {

            $('#install-process').html('<div class="alert alert-primary">Checking package, then installing only if the analyzer marks it safe...</div>');

            let filePath = $(this).data('file-path');
            $.easyAjax({
                type: 'POST',
                url: "{{ route('custom-modules.store') }}",
                blockUI: true,
                data: {
                    "_token": "{{ csrf_token() }}",
                    filePath: filePath,
                    package_attach_mode: $('#package_attach_mode').val(),
                    package_ids: $('#package_ids').val(),
                    seed_company_module_settings: $('#seed_company_module_settings').is(':checked') ? 1 : 0,
                    run_permission_seeders: $('#run_permission_seeders').is(':checked') ? 1 : 0,
                    enable_module_after_install: $('#enable_module_after_install').is(':checked') ? 1 : 0,
                    run_module_migrations: $('#run_module_migrations').is(':checked') ? 1 : 0,
                    run_post_install_audit: $('#run_post_install_audit').is(':checked') ? 1 : 0,
                    auto_repair_metadata: $('#auto_repair_metadata').is(':checked') ? 1 : 0,
                    run_safe_autofix: $('#run_safe_autofix').is(':checked') ? 1 : 0
                },
                success: function (response) {
                    if (response.status === 'success') {
                        const successBanner = '<div class="alert alert-success"><strong>Install complete.</strong> Results stay visible below so you can see exactly what passed, what was repaired, and what still needs attention.</div>';
                        $('#install-process').html(successBanner + (response.diagnostics ? response.diagnostics : `<div class="alert alert-success">@lang('messages.customModuleInstalled')</div>`));
                        $('html, body').animate({ scrollTop: $('#install-process').offset().top - 40 }, 250);
                    }

                    if (response.status === 'fail') {
                        if (response.diagnostics) {
                            $('#install-process').html(response.diagnostics);
                        }
                        else {
                            $('#install-process').html(`<div class="alert alert-danger">${response.message}</div>`);
                        }
                    }
                }
            });
        });

        $(document).on('click', '.copy-analysis-json', function () {
            const payload = $(this).data('json-report');

            if (navigator.clipboard && window.isSecureContext) {
                navigator.clipboard.writeText(payload).then(function () {
                    Swal.fire({icon: 'success', title: 'Copied', text: 'Analysis JSON copied to clipboard.', timer: 1400, showConfirmButton: false});
                });
            }
            else {
                const temp = $('<textarea>');
                $('body').append(temp);
                temp.val(payload).select();
                document.execCommand('copy');
                temp.remove();
                Swal.fire({icon: 'success', title: 'Copied', text: 'Analysis JSON copied to clipboard.', timer: 1400, showConfirmButton: false});
            }
        });

        $(document).on('click', '.download-analysis-json', function () {
            const payload = $(this).data('json-report');
            const fileName = $(this).data('download-name') || 'analysis.json';
            const blob = new Blob([payload], { type: 'application/json;charset=utf-8' });
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = fileName;
            document.body.appendChild(link);
            link.click();
            link.remove();
            URL.revokeObjectURL(url);
        });

        $('.export-analysis').click(function () {
            $('#install-process').html('<div class="alert alert-primary">Generating downloadable JSON analysis...</div>');

            let filePath = $(this).data('file-path');
            $.ajax({
                type: 'POST',
                url: "{{ route('custom-modules.analyze_download') }}",
                data: {
                    "_token": "{{ csrf_token() }}",
                    filePath: filePath
                },
                xhrFields: { responseType: 'blob' },
                success: function (blob, status, xhr) {
                    const disposition = xhr.getResponseHeader('Content-Disposition') || '';
                    const match = disposition.match(/filename="?([^";]+)"?/);
                    const fileName = match ? match[1] : 'module-analysis.json';
                    const url = URL.createObjectURL(blob);
                    const link = document.createElement('a');
                    link.href = url;
                    link.download = fileName;
                    document.body.appendChild(link);
                    link.click();
                    link.remove();
                    URL.revokeObjectURL(url);
                    $('#install-process').html('<div class="alert alert-success">JSON analysis downloaded.</div>');
                },
                error: function (xhr) {
                    const reader = new FileReader();
                    reader.onload = function () {
                        try {
                            const response = JSON.parse(reader.result);
                            if (response && response.diagnostics) {
                                $('#install-process').html(response.diagnostics);
                                return;
                            }
                            $('#install-process').html('<div class="alert alert-danger">Unable to export analysis JSON.</div>');
                        }
                        catch (e) {
                            $('#install-process').html('<div class="alert alert-danger">Unable to export analysis JSON.</div>');
                        }
                    };

                    if (xhr.response) {
                        reader.readAsText(xhr.response);
                    }
                    else {
                        $('#install-process').html('<div class="alert alert-danger">Unable to export analysis JSON.</div>');
                    }
                }
            });
        });

        $('.autofix-files').click(function () {

            $('#install-process').html('<div class="alert alert-primary">Analyzing module and applying safe autofixes...</div>');

            let filePath = $(this).data('file-path');
            $.easyAjax({
                type: 'POST',
                url: "{{ route('custom-modules.apply_safe_fixes') }}",
                blockUI: true,
                data: {
                    "_token": "{{ csrf_token() }}",
                    filePath: filePath,
                    package_attach_mode: $('#package_attach_mode').val(),
                    package_ids: $('#package_ids').val(),
                    auto_repair_metadata: $('#auto_repair_metadata').is(':checked') ? 1 : 0,
                    run_safe_autofix: $('#run_safe_autofix').is(':checked') ? 1 : 0
                },
                success: function (response) {
                    if (response.diagnostics) {
                        const banner = `<div class="alert alert-success"><strong>Safe autofix finished.</strong> The updated analysis and repair results are shown below.</div>`;
                        $('#install-process').html(banner + response.diagnostics);
                    }
                    else {
                        $('#install-process').html(`<div class="alert alert-${response.status === 'success' ? 'success' : 'danger'}">${response.message || 'Safe autofix completed.'}</div>`);
                    }
                    $('html, body').animate({ scrollTop: $('#install-process').offset().top - 40 }, 250);
                }
            });
        });


        $('.delete-files').click(function () {
            let filePath = $(this).data('file-path');
            let fileNumber = $(this).data('file-no');

            Swal.fire({
                title: "@lang('messages.sweetAlertTitle')",
                text: "@lang('messages.removeFileText')",
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
                    $.easyAjax({
                        type: 'POST',
                        url: "{{ route('update-settings.deleteFile') }}",
                        blockUI: true,
                        data: {
                            "_token": "{{ csrf_token() }}",
                            filePath: filePath
                        },
                        success: function (response) {
                            $('#file-' + fileNumber).remove();
                        }
                    });
                }
            });


        });

    </script>
@endpush

