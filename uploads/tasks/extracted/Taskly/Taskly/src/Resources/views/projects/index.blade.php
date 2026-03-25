@extends('layouts.main')
@section('page-title')
    {{ __('Manage Projects') }}
@endsection
@section('page-breadcrumb')
    {{ __('Projects') }}
@endsection
@push('css')
    <link rel="stylesheet" href="{{ asset('packages/workdo/Taskly/src/Resources/assets/css/custom.css') }}" type="text/css" />
@endpush
@section('page-action')
    <div class="d-flex">
        @stack('project_template_button')
        @permission('project import')
            <a href="javascript:void(0)" class="btn btn-sm btn-primary me-2" data-ajax-popup="true"
                data-title="{{ __('Project Import') }}" data-url="{{ route('project.file.import') }}" data-bs-toggle="tooltip"
                title="{{ __('Import') }}"><i class="ti ti-file-import"></i>
            </a>
        @endpermission
        <a href="{{ route('projects.list') }}" class="btn btn-sm btn-primary me-2" data-bs-toggle="tooltip"
            title="{{ __('List View') }}">
            <i class="ti ti-list text-white"></i>
        </a>
        @permission('project create')
            <a class="btn btn-sm btn-primary me-2" data-ajax-popup="true" data-size="md"
                data-title="{{ __('Create Project') }}" data-url="{{ route('projects.create') }}" data-bs-toggle="tooltip"
                title="{{ __('Create') }}">
                <i class="ti ti-plus"></i>
            </a>
        @endpermission
    </div>
@endsection
@section('content')
    <section class="section">
        <div class="row ">
            <div class="col-xl-12 col-lg-12 col-md-12 d-flex align-items-center justify-content-end">
                <div class="text-sm-right status-filter">
                    <div class="btn-group mb-3">
                        <button type="button" class="btn btn-light  text-white btn_tab  bg-primary active"
                            data-filter="All" data-status="All">{{ __('All') }}</button>
                        <button type="button" class="btn btn-light bg-primary text-white btn_tab"
                            data-filter="Ongoing">{{ __('Ongoing') }}</button>
                        <button type="button" class="btn btn-light bg-primary text-white btn_tab"
                            data-filter="Finished">{{ __('Finished') }}</button>
                        <button type="button" class="btn btn-light bg-primary text-white btn_tab"
                            data-filter="OnHold">{{ __('OnHold') }}</button>
                    </div>
                </div>
            </div><!-- end col-->
        </div>

        <div id="multiCollapseExample1">
            <div class="card">
                <div class="card-body">
                    {{ Form::open(['route' => ['projects.index'], 'method' => 'GET', 'id' => 'project_submit']) }}
                    <div class="row d-flex align-items-center justify-content-end">
                        <div class="col-xl-2 col-lg-3 col-md-6 col-sm-12 col-12 mr-2">
                            <div class="btn-box">
                                {{ Form::label('start_date', __('Start Date'), ['class' => 'form-label']) }}
                                {{ Form::date('start_date', isset($_GET['start_date']) ? $_GET['start_date'] : null, ['class' => 'form-control ', 'placeholder' => 'Select Date']) }}

                            </div>
                        </div>
                        <div class="col-xl-2 col-lg-3 col-md-6 col-sm-12 col-12 mr-2">
                            <div class="btn-box">
                                {{ Form::label('end_date', __('End Date'), ['class' => 'form-label']) }}
                                {{ Form::date('end_date', isset($_GET['end_date']) ? $_GET['end_date'] : null, ['class' => 'form-control ', 'placeholder' => 'Select Date']) }}

                            </div>
                        </div>
                        <div class="col-auto float-end mt-4 d-flex">

                            <a href="javascript:void(0)" class="btn btn-sm btn-primary me-2"
                                onclick="document.getElementById('project_submit').submit(); return false;"
                                data-bs-toggle="tooltip" title="{{ __('Apply') }}"
                                data-original-title="{{ __('apply') }}">
                                <span class="btn-inner--icon"><i class="ti ti-search"></i></span>
                            </a>
                            <a href="{{ route('projects.index') }}" class="btn btn-sm btn-danger" data-toggle="tooltip"
                                data-original-title="{{ __('Reset') }}">
                                <span class="btn-inner--icon"><i class="ti ti-trash-off text-white-off"></i></span>
                            </a>
                        </div>
                    </div>
                    {{ Form::close() }}
                </div>
            </div>
        </div>

        <div class="filters-content">
            <div class="row mb-4 project-wrp d-flex">
                @isset($projects)
                    @foreach ($projects as $project)
                        <div class="col-xxl-3 col-xl-4 col-md-6 col-12 All {{ $project->status }}">
                            <div class="project-card">
                                <div class="project-card-inner">
                                    <div class="project-card-header d-flex justify-content-between">
                                        @if ($project->status == 'Finished')
                                            <p class="badge bg-success mb-0 d-flex align-items-center">{{ __('Finished') }}</p>
                                        @elseif($project->status == 'Ongoing')
                                            <p class="badge bg-secondary mb-0 d-flex align-items-center">{{ __('Ongoing') }}
                                            </p>
                                        @else
                                            <p class="badge bg-warning mb-0 d-flex align-items-center">{{ __('OnHold') }}</p>
                                        @endif
                                        @if ($project->is_active)
                                            <button type="button"
                                                class="btn btn-light dropdown-toggle d-flex align-items-center justify-content-center"
                                                data-bs-toggle="dropdown" aria-haspopup="true" aria-expanded="false">
                                                <i class="ti ti-dots-vertical text-black"></i>
                                            </button>

                                            <div class="dropdown-menu dropdown-menu-end pointer">
                                                @permission('project invite user')
                                                    <a href="#!" data-ajax-popup="true" data-size="md"
                                                        data-title="{{ __('Invite Users') }}"
                                                        data-url="{{ route('projects.invite.popup', [$project->id]) }}"
                                                        class="dropdown-item" tabindex="0"><i
                                                            class="ti ti-user-plus me-1"></i><span>{{ __('Invite Users') }}</span></a>
                                                @endpermission

                                                @permission('project manage')
                                                    <a class="dropdown-item" data-ajax-popup="true" data-size="md"
                                                        data-title="{{ __('Share to Clients') }}"
                                                        data-url="{{ route('projects.share.popup', [$project->id]) }}">
                                                        <i class="ti ti-share me-1"></i> <span>{{ __('Share to Clients') }}</span>
                                                    </a>
                                                @endpermission
                                                @permission('project create')
                                                    <a class="dropdown-item" data-ajax-popup="true" data-size="md"
                                                        data-title="{{ __('Duplicate Project') }}"
                                                        data-url="{{ route('project.copy', [$project->id]) }}">
                                                        <i class="ti ti-copy me-1"></i> <span>{{ __('Duplicate') }}</span>
                                                    </a>
                                                @endpermission
                                                @if (module_is_active('ProjectTemplate'))
                                                    @permission('project template create')
                                                        <a class="dropdown-item" data-ajax-popup="true" data-size="md"
                                                            data-title="{{ __('Save As Template') }}"
                                                            data-url="{{ route('project-template.create', ['project_id' => $project->id, 'type' => 'template']) }}">
                                                            <i class="ti ti-bookmark me-1"></i>
                                                            <span>{{ __('Save as template') }}</span>
                                                        </a>
                                                    @endpermission
                                                @endif
                                                @permission('project edit')
                                                    <a class="dropdown-item" data-ajax-popup="true" data-size="lg"
                                                        data-title="{{ __('Edit Project') }}"
                                                        data-url="{{ route('projects.edit', [$project->id]) }}">
                                                        <i class="ti ti-pencil me-1"></i> <span>{{ __('Edit') }}</span>
                                                    </a>
                                                @endpermission
                                                @permission('project delete')
                                                    <form id="delete-form-{{ $project->id }}"
                                                        action="{{ route('projects.destroy', [$project->id]) }}" method="POST">
                                                        @csrf
                                                        <a href="javascript:void(0)"
                                                            class="dropdown-item text-danger delete-popup bs-pass-para show_confirm"
                                                            data-confirm="{{ __('Are You Sure?') }}"
                                                            data-text="{{ __('This action can not be undone. Do you want to continue?') }}"
                                                            data-confirm-yes="delete-form-{{ $project->id }}">
                                                            <i class="ti ti-trash me-1"></i> <span>{{ __('Delete') }}</span>
                                                        </a>
                                                        @method('DELETE')
                                                    </form>
                                                @endpermission
                                            </div>
                                        @else
                                            <div class="btn">
                                                <i class="ti ti-lock"></i>
                                            </div>
                                        @endif
                                    </div>
                                    <div class="project-card-content">
                                        <div class="project-content-top">
                                            <div class="user-info  d-flex align-items-center">
                                                @if ($project->is_active)
                                                    <a href="@permission('project manage') {{ route('projects.show', [$project->id]) }} @endpermission"
                                                        class="wid-30 me-2 border-1 border border-primary rounded-circle"
                                                        tabindex="0">
                                                        <img alt="{{ $project->name }}" class="img-fluid  fix_img"
                                                            avatar="{{ $project->name }}">
                                                    </a>
                                                @else
                                                    <a href="javascript:void(0)"
                                                        class="wid-30 me-2 border-1 border border-primary rounded-circle"
                                                        tabindex="0">
                                                        <img alt="{{ $project->name }}" class="img-fluid  fix_img"
                                                            avatar="{{ $project->name }}">
                                                    </a>
                                                @endif
                                                <h2 class="h5 mb-0">
                                                    <a href="@permission('project manage') {{ route('projects.show', [$project->id]) }} @endpermission"
                                                        tabindex="0" title="{{ $project->name }}"
                                                        class="">{{ $project->name }}</a>
                                                </h2>
                                            </div>
                                            <p>{{ $project->description }}</p>
                                            <div class="d-flex gap-2 align-items-center justify-content-between">
                                                <p class="mb-0"><b>{{ __('Due Date') }} : </b>{{ $project->end_date }}</p>
                                                <div class="view-btn d-flex gap-2 align-items-center">
                                                    @if ($project->is_active)
                                                        <a class="btn btn-warning" data-bs-toggle="tooltip"
                                                            href="{{ route('projects.show', [$project->id]) }}"
                                                            data-bs-original-title="{{ __('View') }}">
                                                            <i class="ti ti-eye text-white"></i>
                                                        </a>
                                                        <a class="btn btn-primary" data-bs-toggle="tooltip"
                                                            href="{{ route('projects.task.board', [$project->id]) }}"
                                                            data-bs-original-title="{{ __('Task Board') }}">
                                                            <i class="ti ti-list-check"></i>
                                                        </a>
                                                    @endif
                                                </div>
                                            </div>
                                        </div>
                                        <div
                                            class="project-content-bottom d-flex align-items-center justify-content-between gap-2">
                                            <div class="d-flex align-items-center gap-2 user-image">
                                                @foreach ($project->users as $user)
                                                    @if ($user->pivot->is_active)
                                                        <img alt="image" data-bs-toggle="tooltip" data-bs-placement="top"
                                                            title="{{ $user->name }}"
                                                            src="{{ $user->avatar ? get_file($user->avatar) : get_file('avatar.png') }}"
                                                            class="border-1 border border-white rounded-circle">
                                                    @endif
                                                @endforeach
                                                <span class="">{{ __('Members') }}</span>
                                            </div>
                                            <div class="comment d-flex align-items-center gap-2">
                                                <p class="d-flex align-items-center gap-1" data-bs-toggle="tooltip"
                                                    data-bs-placement="top" title="{{ __('Tasks') }}"> <i
                                                        class="ti ti-message-dots"></i>
                                                    <span>{{ $project->countTask() }}</span>
                                                </p>
                                                <p class="d-flex align-items-center gap-1" data-bs-toggle="tooltip"
                                                    data-bs-placement="top" title="{{ __('Comments') }}">
                                                    <i class="ti ti-file-invoice"></i>
                                                    <span>{{ $project->countTaskComments() }}</span>
                                                </p>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    @endforeach
                @endisset
                @auth('web')
                    @permission('project create')
                        <div class="col-xxl-3 col-xl-4 col-md-6 col-12 All Ongoing Finished OnHold">
                            <div class="project-card-inner">
                                <a href="javascript:void(0)" class="btn-addnew-project " data-ajax-popup="true" data-size="md"
                                    data-title="{{ __('Create Project') }}" data-url="{{ route('projects.create') }}">
                                    <div class="bg-primary proj-add-icon">
                                        <i class="ti ti-plus"></i>
                                    </div>
                                    <h6 class="mt-4 mb-2">{{ __('Add Project') }}</h6>
                                    <p class="text-muted text-center mb-0">{{ __('Click here to add New Project') }}</p>
                                </a>
                            </div>
                        </div>
                    @endpermission
                @endauth
            </div>

            {!! $projects->links('vendor.pagination.global-pagination') !!}
        </div>
    </section>
@endsection



@push('scripts')
    <script src="{{ asset('packages/workdo/Taskly/src/Resources/assets/js/isotope.pkgd.min.js') }}"></script>

    <script src="{{ asset('js/letter.avatar.js') }}"></script>

    <script>
        $(document).ready(function() {

            $('.status-filter button').click(function() {
                $('.status-filter button').removeClass('active');
                $(this).addClass('active');
                var classAttr = $(this).data('filter');
                if (classAttr === 'All') {
                    // $('.All').removeClass('d-none');
                    $('.All').removeClass('d-none').css('opacity', 1);
                    $('.All').each(function() {
                        $(this).css('transform', 'translateX(0)');
                    });
                } else {
                    // $('.All').addClass('d-none');
                    // $('.' + classAttr).removeClass('d-none');
                    $('.All').addClass('d-none').css('opacity', 0).css('transform', 'translateX(-20px)');
                    $('.' + classAttr).removeClass('d-none').css('opacity', 1).css('transform', 'translateX(0)');
                }

            });

            // Check if the direction is RTL, then set right based on a repeating pattern
            if ($('html').attr('dir') === 'rtl') {
                var $allItems = $('.filters-content .All');
                $allItems.each(function(index) {
                    // Set right property based on a repeating pattern
                    $(this).css('right', (index % 4) * 25 + '%');
                });
            }
        });
    </script>
@endpush
