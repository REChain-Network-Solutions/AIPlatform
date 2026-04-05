@includeIf('module-settings.partials.doctor-summary')
@extends('layouts.app')

@section('content')

    <!-- SETTINGS START -->
    <div class="w-100 d-flex ">

        {{-- SAAS --}}
        @if(user()->is_superadmin)
            <x-super-admin.setting-sidebar :activeMenu="$activeSettingMenu"/>
        @else
            <x-setting-sidebar :activeMenu="$activeSettingMenu"/>
        @endif

        <x-setting-card>
            <x-slot name="header">
                <div class="s-b-n-header" id="tabs">
                    <nav class="tabs px-4 border-bottom-grey">
                        <div class="nav" id="nav-tab" role="tablist">
                            @if (!user()->is_superadmin)
                                <a class="nav-item nav-link f-15 active admin"
                                    href="{{ route('module-settings.index') }}?tab=admin" role="tab"
                                    aria-controls="nav-ticketAgents" aria-selected="true">@lang('app.admin')
                                </a>

                                <a class="nav-item nav-link f-15 employee"
                                    href="{{ route('module-settings.index') }}?tab=employee" role="tab"
                                    aria-controls="nav-ticketTypes" aria-selected="true">@lang('app.employee')
                                </a>

                                <a class="nav-item nav-link f-15 client"
                                    href="{{ route('module-settings.index') }}?tab=client" role="tab"
                                    aria-controls="nav-ticketChannel" aria-selected="true">@lang('app.client')
                                </a>
                            @endif
                            @if(user()->is_superadmin)
                                <a class="nav-item nav-link f-15 custom" href="{{ route('custom-modules.index') }}?tab=custom"
                                   role="tab" aria-controls="nav-ticketChannel"
                                   aria-selected="true">@lang('app.menu.customModule')
                                </a>
                            @endif

                        </div>
                    </nav>
                </div>
            </x-slot>
            @if (user()->is_superadmin)
                <x-slot name="buttons">
                    <div class="row">

                        <div class="col-md-12 my-2">
                            <x-forms.link-primary :link="route('custom-modules.create')" icon="cog">
                                @lang('app.install')/@lang('app.update')
                                @lang('app.module')
                            </x-forms.link-primary>
                        </div>

                    </div>
                </x-slot>
            @endif

            {{-- include tabs here --}}
            @include($view)

        </x-setting-card>

    </div>
    <!-- SETTINGS END -->

@endsection

@push('scripts')
    <script>
        /* manage menu active class */
        $('.nav-item').removeClass('active');
        const activeTab = "{{ $activeTab }}";
        $('.' + activeTab).addClass('active');

        showBtn(activeTab);

        function showBtn(activeTab) {
            $('.actionBtn').addClass('d-none');
            $('.' + activeTab + '-btn').removeClass('d-none');
        }

        $("body").on("click", "#editSettings .nav a", function(event) {
            event.preventDefault();

            $('.nav-item').removeClass('active');
            $(this).addClass('active');

            const requestUrl = this.href;

            $.easyAjax({
                url: requestUrl,
                blockUI: true,
                container: "#nav-tabContent",
                historyPush: true,
                success: function(response) {
                    if (response.status == "success") {
                        showBtn(response.activeTab);

                        $('#nav-tabContent').html(response.html);
                        init('#nav-tabContent');
                    }
                }
            });
        });


        /* change module setting */
        $('body').on('change', '.change-module-setting', function() {

            var id = $(this).data('setting-id');
            var name = $(this).data('module-name');

            if(name == 'settings') {
                Swal.fire({
                    icon: 'error',
                    text: '@lang("messages.settingModuleCannotBeDisabled")',
                    toast: true,
                    position: 'top-end',
                    timer: 3000,
                    timerProgressBar: true,
                    showConfirmButton: false,
                    customClass: {
                        confirmButton: 'btn btn-primary',
                    },
                    showClass: {
                        popup: 'swal2-noanimation',
                        backdrop: 'swal2-noanimation'
                    },
                })

                $(this).prop('checked', true);
                return false;
            }


            var moduleStatus = $(this).is(':checked') ? 'active' : 'deactive';
            var token = '{{ csrf_token() }}';
            var url = "{{ route('module-settings.update', ':id') }}";
            url = url.replace(':id', id);

            $.easyAjax({
                type: 'POST',
                url: url,
                container: '.settings-box',
                blockUI: true,
                data: {
                    '_token': token,
                    'status': moduleStatus,
                    'name': name,
                    '_method':'PUT'
                },
                success: function () {
                    window.location.reload();
                }
            });
        });

    </script>
@endpush

@includeIf('module-settings.partials.menu-truth-summary')
