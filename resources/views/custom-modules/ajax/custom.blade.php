<div class="table-responsive p-20">

    <div id="update-area" class="mt-20 mb-20 col-md-12 white-box d-none">
        {{__('app.loading')}}
    </div>
    <div class="alert alert-danger d-none" id="custom-module-alert"></div>
    {{-- SAAS   --}}
    @if(session('subdomain_module_activated') == 'activated')
        <div class="alert bg-light-warning border-0 rounded-lg shadow-sm p-4 mb-3">
            <div class="d-flex">
                <div class="mr-3 pt-1">
                    <i class="fas fa-shield-alt text-warning f-24"></i>
                </div>
                <div class="flex-grow-1">
                    <h4 class="text-warning font-weight-bold mb-3">Security Configuration Required</h4>

                    <div class="bg-white rounded p-3 mb-3">
                        <h6 class="font-weight-bold mb-2">Required Actions:</h6>
                        <ul class="list-unstyled mb-0">
                            <li class="d-flex align-items-start mb-2">
                                <i class="fas fa-check-circle text-success mr-2 mt-1"></i>
                                <div class="d-flex align-items-center flex-wrap">
                                    <span class="mr-2">Configure wildcard subdomains on your server</span>
                                    <a href="https://www.youtube.com/watch?v=0KOHj4a2Sek" class="btn btn-sm btn-warning" target="_blank">
                                        <i class="fab fa-youtube mr-1"></i> Watch Guide
                                    </a>
                                </div>
                            </li>
                            <li class="d-flex align-items-center">
                                <i class="fas fa-check-circle text-success mr-2"></i>
                                <span>Update DNS settings appropriately</span>
                            </li>
                        </ul>
                    </div>

                    <div class="bg-white rounded p-3">
                        <h6 class="font-weight-bold mb-2">Important Changes:</h6>
                        <ul class="list-unstyled mb-0">
                            <li class="d-flex align-items-center mb-2">
                                <i class="fas fa-link text-primary mr-2"></i>
                                <div>
                                    <span class="mr-2">New Superadmin Login:</span>
                                    <code class="bg-light-primary px-2 py-1 rounded">{{ url('/') }}/super-admin-login</code>
                                </div>
                            </li>
                            <li class="d-flex align-items-center mb-2">
                                <i class="fas fa-lock text-danger mr-2"></i>
                                <span>Public login page is now disabled</span>
                            </li>
                            <li class="d-flex align-items-center">
                                <i class="fas fa-building text-success mr-2"></i>
                                <span>Companies now have dedicated subdomain login pages</span>
                            </li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    @endif

    @includeIf('languagepack::module-activated-alert')

    @include('custom-modules.sections.universal-bundle')

    <x-table class="table-bordered table-hover custom-modules-table" headType="thead-light">
        <x-slot name="thead">
            <th>@lang('app.name')</th>
            @if (!$universalBundle)
            <th>@lang('app.purchaseCode')</th>
            @endif
            <th>@lang('app.moduleVersion')</th>
            @if (!$universalBundle)
            <th class="text-right">@lang('app.notify')</th>
            @endif
            <th class="text-right">@lang('app.status')</th>
        </x-slot>

        @forelse ($allModules as $key=>$module)
        @php
            $fetchSetting = null;
            if (in_array($module, $worksuitePlugins) && config(strtolower($module) . '.setting'))
            {
                $fetchSetting = config(strtolower($module) . '.setting')::first();
            }
        @endphp
            <tr>
                <td><span>{{ $key }}</span>
                    @if (module_enabled('UniversalBundle') && isInstallFromUniversalBundleModule($key))
                    
                            <i class="icon text-info fas fa-info-circle cursor-pointer" data-toggle="tooltip"
                              data-original-title="{{__('universalbundle::app.moduleInfo')}}"></i>
                    @else
{{--                        @if ($fetchSetting?->purchase_code && $fetchSetting?->supported_until)--}}
{{--                            <i class="icon text-info fas fa-info-circle cursor-pointer"--}}
{{--                            data-toggle="popover" data-placement="top" data-html="true" data-trigger="hover"--}}
{{--                            data-content="@include('custom-modules.sections.support-date')"></i>--}}
{{--                        @endif--}}
                    @endif
{{--                    @if ($fetchSetting?->license_type && !(module_enabled('UniversalBundle') && isInstallFromUniversalBundleModule($key)))--}}
{{--                        <span class="ml-2 badge badge-secondary">{{ $fetchSetting->license_type }}</span>--}}
{{--                        @if(str_contains($fetchSetting->license_type, 'Regular'))--}}
{{--                            <a href="{{ \Froiden\Envato\Helpers\FroidenApp::buyExtendedUrl(config(strtolower($module) . '.envato_item_id')) }}"--}}
{{--                            target="_blank">Upgrade now</a>--}}
{{--                        @endif--}}
{{--                    @endif--}}
                </td>
                @if (!$universalBundle)
                <td>
                    @if ($fetchSetting)
                        @if (config(strtolower($module) . '.verification_required'))
                            @include('custom-modules.sections.purchase-code')
                        @endif
                    @endif
                </td>
                @endif
                <td>
                    @if (config(strtolower($module) . '.setting'))
                        @include('custom-modules.sections.version')

                        @if ($plugins->where('envato_id', config(strtolower($module) . '.envato_item_id'))->first() && !(module_enabled('UniversalBundle') && isInstallFromUniversalBundleModule($key)))
                            @include('custom-modules.sections.module-update')
                        @endif
                    @endif

                </td>

                @if (!$universalBundle)

                <td class="text-right">
                    @if ($fetchSetting)
                    <div class="custom-control custom-switch ml-2 d-inline-block"  data-toggle="tooltip"
                         data-original-title="@lang('app.moduleNotifySwitchMessage', ['name' => $module])">
                        <input type="checkbox" class="custom-control-input change-module-notification"
                                @checked($fetchSetting->notify_update)
                               id="module-notification-{{ $key }}" data-module-name="{{ $module }}">
                        <label class="custom-control-label cursor-pointer" for="module-notification-{{ $key }}"></label>
                    </div>
                    @endif
                </td>
                @endif

                <td class="text-right">
                    <div class="custom-control custom-switch ml-2 d-inline-block"  data-toggle="tooltip"
                         data-original-title="@lang('app.moduleSwitchMessage', ['name' => $module])">
                        <input type="checkbox" @if (in_array($module, $worksuitePlugins)) checked
                               @endif class="custom-control-input change-module-status"
                               id="module-{{ $key }}" data-module-name="{{ $module }}">
                        <label class="custom-control-label cursor-pointer" for="module-{{ $key }}"></label>
                    </div>
                </td>
            </tr>
        @empty
            <tr>
                <td colspan="5">
                    <x-cards.no-record icon="calendar" :message="__('messages.noRecordFound')"/>
                </td>
            </tr>
        @endforelse

    </x-table>

    @include('vendor.froiden-envato.update.plugins', ['allModules' => $allModules])
</div>

<script>
    $('body').on('change', '.change-module-status', function () {
        let moduleStatus;
        const module = $(this).data('module-name');

        if ($(this).is(':checked')) {
            moduleStatus = 'active';

            if(module === 'Subdomain') {
                Swal.fire({
                    title: '<i class="fas fa-exclamation-triangle text-warning"></i> Important Configuration Required',
                    html: `
                        <div class="text-left">
                            <div class="alert alert-warning mb-3" style="line-height: 1.6;">
                                Please ensure you have properly configured wildcard subdomains on your server before proceeding.
                                <div class="mt-2">
                                    <a href="https://www.youtube.com/watch?v=0KOHj4a2Sek" class="btn btn-sm btn-warning" target="_blank">
                                        <i class="fab fa-youtube mr-1"></i> Watch Configuration Guide
                                    </a>
                                </div>
                            </div>

                            <div class="card border mb-3">
                                <div class="card-header bg-light">
                                    <strong>Changes After Activation:</strong>
                                </div>
                                <div class="card-body">
                                    <ul class="mb-0 pl-3" style="line-height: 1.8;">
                                        <li class="mb-2">New Superadmin Login URL:<br>
                                            <code class="bg-light px-2 py-1 d-inline-block mt-1">${window.location.origin}/super-admin-login</code>
                                        </li>
                                        <li class="mb-2">Public login page will be disabled</li>
                                        <li>Each company will have a dedicated login page on their subdomain</li>
                                    </ul>
                                </div>
                            </div>

                            <p class="mb-0" style="line-height: 1.6;">Are you sure you want to proceed with activation?</p>
                        </div>
                    `,
                    showCancelButton: true,
                    confirmButtonText: '<i class="fas fa-check-circle"></i> Yes, activate',
                    cancelButtonText: '<i class="fas fa-times-circle"></i> No, cancel',
                    confirmButtonColor: '#0d6efd', // Changed to Bootstrap primary blue
                    cancelButtonColor: '#6c757d', // Changed to Bootstrap gray
                    customClass: {
                        confirmButton: 'btn btn-primary ml-2 mr-2', // Added ml-2 to move activate button to right
                        cancelButton: 'btn btn-secondary'  // Changed to secondary style
                    },
                    buttonsStyling: false
                }).then((result) => {
                    if (!result.isConfirmed) {
                        $('#module-' + module).prop('checked', false);
                        return;
                    }
                    updateModuleStatus(module, moduleStatus);
                });
                return;
            }
        } else {
            moduleStatus = 'inactive';
        }

        updateModuleStatus(module, moduleStatus);
    });

    function updateModuleStatus(module, moduleStatus) {
        let url = "{{ route('custom-modules.update', ':module') }}";
        url = url.replace(':module', module);

        $('#custom-module-alert').addClass('d-none');

        $.easyAjax({
            url: url,
            type: "POST",
            disableButton: true,
            buttonSelector: ".change-module-status",
            container: '.custom-modules-table',
            blockUI: true,
            data: {
                'id': module,
                'status': moduleStatus,
                '_method': 'PUT',
                '_token': '{{ csrf_token() }}'
            },
            error: function (response) {
                if (response.responseJSON) {
                    $('#custom-module-alert').html(response.responseJSON.message).removeClass('d-none');
                    $('#module-' + module).prop('checked', false);
                }
            }
        });
    }

    $('body').on('click', '.verify-module', function () {
        const module = $(this).data('module');
        let url = "{{ route('custom-modules.show', ':module') }}";
        url = url.replace(':module', module);
        $(MODAL_LG + ' ' + MODAL_HEADING).html('...');
        $.ajaxModal(MODAL_LG, url);
    });

</script>
@includeIf('vendor.froiden-envato.update.update_module')
