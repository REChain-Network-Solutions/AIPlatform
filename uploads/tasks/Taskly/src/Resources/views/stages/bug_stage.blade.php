
@extends('layouts.main')
@section('page-title')
    {{__('Manage Bug Stages')}}
@endsection


@section('page-action')
@endsection

@section('content')
<div id="bug-stages-settings" class="row">
    <div class="col-md-3">
        @include('taskly::layouts.system_setup')
    </div>
    <div class="col-md-9">
        <div class="card bug-stages" data-value="{{ json_encode($bugStages) }}">
            <div class="card-header">
                <div class="row">
                    <div class="col-11">
                        <h5 class="">
                            {{ __('Bug Stages') }}

                        </h5>
                        <small
                            class="">{{ __('System will consider the last stage as a completed / done project or bug status.') }}</small>
                    </div>
                    @permission('bugstage manage')
                        <div class=" col-1 text-end">
                            <button data-repeater-create type="button" class="btn-submit btn btn-sm btn-primary "
                                data-bs-toggle="tooltip" title="{{ __('Add') }}">
                                <i class="ti ti-plus"></i>
                            </button>
                        </div>
                    @endpermission
                </div>
            </div>
            <div class="card-body">
                <form method="post" action="{{ route('bugstages.store') }}">
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
                            @permission('bugstage delete')
                                <th class="text-right">{{ __('Delete') }}</th>
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
                                    <input type="text" name="name" class="form-control mb-0"
                                        @if (!auth()->user()->isAbleTo('bugstage edit')) readonly @endif required />
                                </td>
                                @permission('bugstage delete')
                                    <td class="text-right">
                                        <a data-repeater-delete
                                            class="action-btn btn-danger  btn btn-sm d-inline-flex align-items-center"
                                            data-toggle="tooltip" title="{{ __('Delete') }}"><i
                                                class="ti ti-trash text-white"></i></a>
                                    </td>
                                @endpermission

                            </tr>
                        </tbody>
                    </table>
                    @permission('bugstage manage')
                    <div class="row">
                        <div class="text-sm col-6 pt-2">
                            {{__('Note : You can easily change order of Bug stage using drag & drop.')}}
                        </div>
                        <div class="text-end col-6 pt-2">
                            <button class="btn-submit btn btn-primary" type="submit">{{ __('Save Changes') }}</button>
                        </div>
                    </div>
                    @endpermission
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
            var $dragAndDropBug = $("body .bug-stages tbody").sortable({
                handle: '.sort-handler'
            });

            var $repeaterBug = $('.bug-stages').repeater({
                initEmpty: true,
                defaultValues: {},
                show: function () {
                    $(this).slideDown();
                },
                hide: function (deleteElement) {
                    var form = $(this).closest("form");
                var title = $(this).attr("data-confirm");
                var text = $(this).attr("data-text");
                if (title == '' || title == undefined) {
                    title = "Are you sure?";

                }
                if (text == '' || text == undefined) {
                    text = "This action can not be undone. Do you want to continue?";

                }
                const swalWithBootstrapButtons = Swal.mixin({
                    customClass: {
                        confirmButton: 'btn btn-success',
                        cancelButton: 'btn btn-danger'
                    },
                    buttonsStyling: false
                })
                swalWithBootstrapButtons.fire({
                    title: title,
                    text: text,
                    icon: 'warning',
                    showCancelButton: true,
                    confirmButtonText: 'Yes',
                    cancelButtonText: 'No',
                    reverseButtons: true
                }).then((result) => {
                    if (result.isConfirmed) {
                            $(this).slideUp(deleteElement);
                        }
                    });
                },
                ready: function (setIndexes) {
                    $dragAndDropBug.on('drop', setIndexes);
                },
                isFirstItemUndeletable: true
            });


            var valuebug = $(".bug-stages").attr('data-value');
            if (typeof valuebug != 'undefined' && valuebug.length != 0){
                valuebug = JSON.parse(valuebug);
                $repeaterBug.setList(valuebug);
            }
        });
    </script>
@endpush

