@extends('layouts.main')
@section('page-title')
    {{__('Manage Task Stages')}}
@endsection


@section('page-action')
@endsection

@section('content')
<div id="task-stages-settings" class="row">
    <div class="col-sm-3">
        @include('taskly::layouts.system_setup')
    </div>
    <div class="col-md-9">
        <div class="card task-stages" data-value="{{ json_encode($stages) }}">
            <div class="card-header">
                <div class="row">
                    <div class="col-11">
                        <h5 class="">
                            {{ __('Task Stages') }}
                        </h5>
                        <small
                            class="">{{ __('System will consider the last stage as a completed / done project or task status.') }}</small>
                    </div>
                    <div class="col-1  text-end">
                        @permission('taskstage manage')
                            <button data-repeater-create type="button"
                                class="btn-submit btn btn-sm btn-primary btn-icon " data-bs-toggle="tooltip"
                                title="{{ __('Add') }}">
                                <i class="ti ti-plus"></i>
                            </button>
                        @endpermission
                    </div>
                </div>
            </div>

            <div class="card-body">
                <form method="post" action="{{ route('stages.store') }}">
                    @csrf
                    <table class="table table-hover" data-repeater-list="stages">
                        <thead>
                            <th>
                                <div data-toggle="tooltip" data-placement="left"
                                    data-title="{{ __('Drag Stage to Change Order') }}" data-original-title=""
                                    title="">
                                    <i class="fas fa-crosshairs"></i>
                                </div>
                            </th>
                            <th>{{ __('Color') }}</th>
                            <th>{{ __('Name') }}</th>
                            @permission('taskstage delete')
                                <th class="text-right">{{ __('Action') }}</th>
                            @endpermission
                        </thead>
                        <tbody>
                            <tr data-repeater-item>
                                <td><i class="fas fa-crosshairs sort-handler"></i></td>
                                <td>
                                    <input type="color" name="color">
                                </td>
                                <td>
                                    <input type="hidden" name="id" id="id" />
                                    <input type="text" name="name" class="form-control mb-0" placeholder="Enter Name"
                                        @if (!auth()->user()->isAbleTo('taskstage edit')) readonly @endif required />
                                </td>
                                @permission('taskstage delete')
                                    <td class="text-right ">
                                        <a data-repeater-delete
                                            class=" action-btn btn-danger  btn btn-sm d-inline-flex align-items-center"
                                            data-toggle="tooltip" title="{{ __('Delete') }}"><i
                                                class="ti ti-trash text-white"></i></a>
                                    </td>
                                @endpermission
                            </tr>
                        </tbody>
                    </table>
                    @permission('taskstage manage')
                    @endpermission
                    <div class="row">
                        <div class="text-sm col-6 pt-2">
                            {{__('Note : You can easily change order of Task stage using drag & drop.')}}
                        </div>
                        <div class="col-6 text-end pt-2">
                            <button class="btn-submit btn btn-primary" type="submit">{{ __('Save Changes') }}</button>
                        </div>
                    </div>
                </form>
            </div>
        </div>
    </div>
</div>

@endsection


@push('scripts')
    <script src="{{ asset('packages/workdo/Taskly/src/Resources/assets/js/jquery-ui.min.js')}}"></script>
    <script src="{{ asset('packages/workdo/Taskly/src/Resources/assets/js/repeater.js')}}"></script>
    <script src="{{ asset('packages/workdo/Taskly/src/Resources/assets/js/colorPick.js')}}"></script>
    <script src="{{asset('assets/js/pages/wow.min.js')}}"></script>
    <script>
    var scrollSpy = new bootstrap.ScrollSpy(document.body, {
            target: '#useradd-sidenav',
            offset: 300
        })
    </script>


    <script>
        $(document).ready(function () {
            var $dragAndDrop = $("body .task-stages tbody").sortable({
                handle: '.sort-handler'
                });

            var $repeater = $('.task-stages').repeater({
                initEmpty: true,
                defaultValues: {},
                show: function () {
                    $(this).slideDown();
                },
                hide: function (deleteElement) {
                    if (confirm('{{__('Are you sure ?')}}')) {
                        $(this).slideUp(deleteElement);
                    }
                },
                ready: function (setIndexes) {
                    $dragAndDrop.on('drop', setIndexes);
                },
                isFirstItemUndeletable: true
            });


            var value = $(".task-stages").attr('data-value');
            if (typeof value != 'undefined' && value.length != 0) {
                value = JSON.parse(value);
                $repeater.setList(value);
            }
            var type = window.location.hash.substr(1);
            $('.list-group-item').removeClass('active');
            $('.list-group-item').removeClass('text-primary');
            if (type != '') {
                $('a[href="#' + type + '"]').addClass('active').removeClass('text-primary');
            } else {
                $('.list-group-item:eq(0)').addClass('active').removeClass('text-primary');
            }
        });
    </script>
@endpush
