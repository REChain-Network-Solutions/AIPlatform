@if ($universalBundle)
    <div class='card border mb-4'>




        <div class="card-body bg-white border-0 pt-4">

            @if ($plugins->where('envato_id', config(strtolower($universalBundle) . '.envato_item_id'))->first())
                @php
                    $currentVersion = \Illuminate\Support\Facades\File::get($universalBundle->getPath() . '/version.txt');
                    $latestVersion = $plugins->where('envato_id', config(strtolower($universalBundle) . '.envato_item_id'))->pluck('version')->first();
                    $needsUpdate = $latestVersion > $currentVersion;
                @endphp
                @if ($needsUpdate)
                    <div class="row pb-4">
                        <div class="col-md-12 text-center">
                            <div class="version-info d-flex justify-content-center">
                                <div class="alert alert-warning w-100 text-center">
                                    <div class="d-flex flex-column align-items-center">
                                        <h5 class="font-weight-bold mb-2">
                                            <i class="fa fa-exclamation-triangle mr-2"></i>
                                            Update Available for {{ $universalBundle->getName() }}
                                        </h5>
                                        <p>
                                            Current version: <span class="badge badge-danger">v{{ $currentVersion }}</span>
                                            &nbsp;|&nbsp;
                                            Latest version: <span class="badge badge-success">v{{ $latestVersion }}</span>
                                        </p>
                                        <p>
                                            @lang('app.moduleUpdateMessage', [
                                                'name' => $universalBundle->getName(),
                                                'version' => $latestVersion
                                            ])
                                        </p>
                                        <div class="mt-2">
                                            @if($plugins->where('envato_id', config(strtolower($universalBundle) . '.envato_item_id'))->pluck('version')->first() > \Illuminate\Support\Facades\File::get($universalBundle->getPath() . '/version.txt'))
                                                <button type="button" class="btn btn-secondary btn-sm update-module"
                                                        data-product-url="{{ Froiden\Envato\Helpers\FroidenApp::renewSupportUrl(config(strtolower($universalBundle) . '.envato_item_id')) }}"
                                                        data-module-name="{{ $universalBundle }}">
                                                    <i class="fa fa-download"></i> @lang('app.update')
                                                </button>
                                            @endif
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                @endif
            @endif

            <div class="row">
                @php
                    $fetchSetting = null;
                    if (in_array($universalBundle, $worksuitePlugins) && config(strtolower($universalBundle) . '.setting')) {
                        $fetchSetting = config(strtolower($universalBundle) . '.setting')::first();
                    }
                @endphp



                <div class="col-md-6">
                    <div class="d-flex">
                        <svg xmlns="http://www.w3.org/2000/svg" width="30" height="30" viewBox="0 0 148.319 148.319">
                            <g id="Group_1192" data-name="Group 1192" transform="translate(0 0)">
                                <path id="Path_1092" data-name="Path 1092" d="M222.36,482h8.691v8.691H222.36Z"
                                    transform="translate(-157.945 -342.371)" fill="#fc6" />
                                <path id="Path_1093" data-name="Path 1093" d="M0,259.64H8.691v8.691H0Z"
                                    transform="translate(0 -184.426)" fill="#ffe666" />
                                <path id="Path_1094" data-name="Path 1094"
                                    d="M87.9,343.049H96.6v8.691H87.9ZM24.383,368.021c10.842-10.485,22.345,1.1,22.345,1.1,30.347-8.212,29.264-38.96,29.264-38.96L64.876,310.85,20.835,363.085A21.736,21.736,0,0,0,24.383,368.021Z"
                                    transform="translate(-14.799 -220.801)" fill="#fc6" />
                                <path id="Path_1095" data-name="Path 1095"
                                    d="M.755,329.786l8.483,5.51c-7.059,8.145-5.2,15.831-2.667,20.6L54.71,307.76,39.5,300.74s-30.53-1.3-38.741,29.046Z"
                                    transform="translate(-0.536 -213.614)" fill="#ffe666" />
                                <path id="Path_1096" data-name="Path 1096"
                                    d="M138.662,330.161l-11.117-19.31-18.483,26.676A32.686,32.686,0,0,0,138.662,330.161Z"
                                    transform="translate(-77.469 -220.802)" fill="#cc295f" />
                                <path id="Path_1097" data-name="Path 1097"
                                    d="M106.7,330.359l22.579-22.579-15.213-7.02A32.686,32.686,0,0,0,106.7,330.359Z"
                                    transform="translate(-75.108 -213.634)" fill="#f37" />
                                <path id="Path_1098" data-name="Path 1098"
                                    d="M172.865,302.977l11.117,11.117,6.553-6.553-11.117-19.31Z"
                                    transform="translate(-122.788 -204.734)" fill="#cfcfe6" />
                                <path id="Path_1099" data-name="Path 1099"
                                    d="M141.043,278.14l-6.552,6.553,11.116,11.117,10.649-10.649Z"
                                    transform="translate(-95.531 -197.567)" fill="#ece6f2" />
                                <path id="Path_1100" data-name="Path 1100"
                                    d="M213.154,284.922l-11.117-19.31-6.552,14.746L206.6,291.475Z"
                                    transform="translate(-138.856 -188.668)" fill="#a3aacc" />
                                <path id="Path_1101" data-name="Path 1101"
                                    d="M163.662,255.521l-6.552,6.552,11.117,11.117,10.649-10.649Z"
                                    transform="translate(-111.597 -181.5)" fill="#cfcfe6" />
                                <path id="Path_1102" data-name="Path 1102"
                                    d="M312.615,257.181c4.392,17.012-20.209,41.613-20.209,41.613L279.1,285.486l14.1-25.1Z"
                                    transform="translate(-198.247 -182.679)" fill="#cc295f" />
                                <path id="Path_1103" data-name="Path 1103"
                                    d="M152.784,115.913c-17.012-4.392-41.613,20.209-41.613,20.209l13.308,13.308,25.1-14.1Z"
                                    transform="translate(-78.966 -81.961)" fill="#f37" />
                                <path id="Path_1104" data-name="Path 1104"
                                    d="M390.679,9.65,345.137,47l29.646,21.452a55.305,55.305,0,0,0,15.9-58.8Z"
                                    transform="translate(-245.156 -6.854)" fill="#cc295f" />
                                <path id="Path_1105" data-name="Path 1105"
                                    d="M299.369,18.692l17.355,25.549L358.17,2.8A55.305,55.305,0,0,0,299.369,18.692Z"
                                    transform="translate(-212.646 0)" fill="#f37" />
                                <path id="Path_1106" data-name="Path 1106"
                                    d="M263.1,138.579l-26.593,18.4-18.4,26.593,17.669,17.669,47.844-40.4q.471-.4.932-.807Z"
                                    transform="translate(-154.922 -98.435)" fill="#ece6f2" />
                                <path id="Path_1107" data-name="Path 1107"
                                    d="M198.32,64.526q-.409.462-.807.932L157.11,113.3l17.669,17.669,44.993-44.993Z"
                                    transform="translate(-111.597 -45.834)" fill="#fff5f5" />
                                <path id="Path_1108" data-name="Path 1108"
                                    d="M357.067,122.224a15.946,15.946,0,0,0,0-22.526l-15.36,7.166-7.166,15.36A15.946,15.946,0,0,0,357.067,122.224Z"
                                    transform="translate(-237.629 -70.817)" fill="#101040" />
                                <path id="Path_1109" data-name="Path 1109"
                                    d="M323.12,88.277a15.946,15.946,0,0,0,0,22.526l22.526-22.526A15.947,15.947,0,0,0,323.12,88.277Z"
                                    transform="translate(-226.208 -59.396)" fill="#283366" />
                                <path id="Path_1110" data-name="Path 1110"
                                    d="M365.991,131.147a7.246,7.246,0,0,0,0-10.236l-7.166,3.07-3.07,7.166A7.245,7.245,0,0,0,365.991,131.147Z"
                                    transform="translate(-252.698 -85.885)" fill="#cfcfe6" />
                                <path id="Path_1111" data-name="Path 1111"
                                    d="M350.564,115.722a7.246,7.246,0,0,0,0,10.236L360.8,115.722A7.246,7.246,0,0,0,350.564,115.722Z"
                                    transform="translate(-247.507 -80.695)" fill="#ece6f2" />
                            </g>
                        </svg>

                        <div class="">
                            <span class="h5 mb-0 mx-2">{{ $universalBundle->getName() }}</span>
                            @if (config(strtolower($universalBundle) . '.setting'))
                                @include('custom-modules.sections.version', ['module' => $universalBundle])
                            @endif
                            <div class="mx-2">
                                @if ($fetchSetting?->license_type)
                                    <span class="badge badge-secondary">{{ $fetchSetting->license_type }}</span>
                                    @if (str_contains($fetchSetting->license_type, 'Regular'))
                                        <a href="{{ \Froiden\Envato\Helpers\FroidenApp::buyExtendedUrl(config(strtolower($universalBundle) . '.envato_item_id')) }}"
                                            target="_blank">Upgrade now</a>
                                    @endif
                                @endif

                                @if (
                                    $fetchSetting?->purchase_code &&
                                        $fetchSetting?->supported_until &&
                                        \Carbon\Carbon::parse($fetchSetting->supported_until)->isPast())
                                    <span class="badge badge-danger">Support Expired</span>
                                @endif

                            </div>
                            @if ($fetchSetting?->purchase_code && $fetchSetting?->supported_until)
                                <div class="mx-2 mt-2">
{{--                                    <i class="fa fa-info-circle"></i>--}}
                                    @include('custom-modules.sections.support-date')
                                </div>
                            @endif
                        </div>
                    </div>
                </div>

                <div class="col-md-6 d-flex align-items-center justify-content-end">
                    <div class="">
                        @if ($fetchSetting)
                            @if (config(strtolower($universalBundle) . '.verification_required'))
                                <div class="d-flex align-items-center justify-content-end">
                                     @include('custom-modules.sections.purchase-code', ['module' => $universalBundle])
                                </div>
                            @endif
                            @if ($plugins->where('envato_id', config(strtolower($universalBundle) . '.envato_item_id'))->first())
                                <div class="mt-3 d-flex align-items-center justify-content-end">
                                    @include('custom-modules.sections.module-update', ['module' => $universalBundle, 'fetchSetting' => $fetchSetting])
                                    {{-- <div class="custom-control custom-switch ml-2 d-inline-block"  data-toggle="tooltip"
                                        data-original-title="@lang('app.moduleNotifySwitchMessage', ['name' => $universalBundle])">
                                        <input type="checkbox" class="custom-control-input change-module-notification"
                                                @checked($fetchSetting->notify_update)
                                            id="module-notification-{{ $universalBundle }}" data-module-name="{{ $universalBundle }}">
                                        <label class="custom-control-label cursor-pointer" for="module-notification-{{ $universalBundle }}"></label>
                                    </div> --}}
                                </div>
                            @endif
                        @endif
                    </div>
                    @if (!config(strtolower($universalBundle) . '.name'))
                        <div class="custom-control custom-switch ml-2" data-toggle="tooltip"
                            data-original-title="@lang('app.moduleSwitchMessage', ['name' => $universalBundle])">
                            <input type="checkbox" class="custom-control-input change-module-status"
                                id="module-{{ $universalBundle }}" data-module-name="{{ $universalBundle }}">
                            <label class="custom-control-label cursor-pointer"
                                for="module-{{ $universalBundle }}"></label>
                        </div>
                    @endif
                </div>
            </div>
        </div>
    </div>
    @includeIf('universalbundle::install-modules')
@endif
