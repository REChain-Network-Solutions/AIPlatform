@extends('layouts.app')
@section('title', __('project::lang.project'))

@section('content')
    @include('project::layouts.nav')
    <section class="content-header">
        <h1 class="tw-text-xl md:tw-text-3xl tw-font-bold tw-text-black" >
            @lang('project::lang.projects')
            <small> @lang('project::lang.all_projects')</small>
        </h1>
    </section>
    <!-- Main content -->
    <section class="content">
        @if ($project_view == 'list_view')
            <div class="row">
                @foreach ($project_stats as $project)
                    <div class="col-md-3 col-sm-6 col-xs-12 col-custom project_stats">
						@component('components.static', [
    'svg' => '<svg xmlns="http://www.w3.org/2000/svg" 
                     width="24"  
                     height="24"  
                     viewBox="0 0 24 24"  
                     fill="none"  
                     stroke="currentColor"  
                     stroke-width="2"  
                     stroke-linecap="round"  
                     stroke-linejoin="round"  
                     class="icon icon-tabler icons-tabler-outline icon-tabler-exclamation-mark {{ $svg_bg }}">
                     <path stroke="none" d="M0 0h24v24H0z" fill="none"/>
                     <path d="M12 19v.01" />
                     <path d="M12 15v-10" />
             </svg>',
    'svg_text' => $project->status == 'not_started' ? 'tw-text-red-500' : 
                  ($project->status == 'on_hold' ? 'tw-text-yellow-500' : 
                  ($project->status == 'cancelled' ? 'tw-text-red-500' : 
                  ($project->status == 'in_progress' ? 'tw-text-cyan-500' : 
                  ($project->status == 'completed' ? 'tw-text-green-500' : '')))),
    'svg_bg' => $project->status == 'not_started' ? 'tw-bg-red-100' : 
                ($project->status == 'on_hold' ? 'tw-bg-yellow-100' : 
                ($project->status == 'cancelled' ? 'tw-bg-red-100' : 
                ($project->status == 'in_progress' ? 'tw-bg-cyan-300' : 
                ($project->status == 'completed' ? 'tw-bg-green-200' : ''))))
])

						                            <p class="tw-text-sm tw-font-medium tw-text-gray-500 tw-truncate tw-whitespace-nowrap">
                                {{ $statuses[$project->status] }}
                            </p>
                            <p
                                class="tw-mt-0.5 tw-text-gray-900 tw-text-xl tw-truncate tw-font-semibold tw-tracking-tight tw-font-mono">
                                {{ $project->count }}
                            </p>
                        @endcomponent
                    </div>
                @endforeach
            </div>
        @endif
        @component('components.widget')
            <div class="box-header with-border">
                <h3 class="box-title">@lang('project::lang.projects')</h3>
                <div class="box-tools pull-right">
                    <div class="btn-group btn-group-toggle" data-toggle="buttons">
                        <label class="btn btn-info btn-sm active list">
                            <input type="radio" name="project_view" value="list_view" class="project_view"
                                data-href="{{ action([\Modules\Project\Http\Controllers\ProjectController::class, 'index']) . '?project_view=list_view' }}">
                            @lang('project::lang.list_view')
                        </label>
                        <label class="btn btn-info btn-sm kanban">
                            <input type="radio" name="project_view" value="kanban" class="project_view"
                                data-href="{{ action([\Modules\Project\Http\Controllers\ProjectController::class, 'index']) . '?project_view=kanban' }}">
                            @lang('project::lang.kanban_board')
                        </label>
                    </div>
                    @can('project.create_project')
                        <button type="button" class="tw-dw-btn tw-dw-btn-primary tw-text-white tw-dw-btn-sm add_new_project"
                            data-href="{{ action([\Modules\Project\Http\Controllers\ProjectController::class, 'create']) }}">
                            @lang('project::lang.new_project')&nbsp;
                            <i class="fa fa-plus"></i>
                        </button>
                    @endcan
                </div>
            </div>
            <div class="box-body">
                <div class="row">
                    @if ($project_view == 'list_view')
                        <div class="col-md-3 project_status_filter">
                            <div class="form-group">
                                {!! Form::label('project_status_filter', __('sale.status') . ':') !!}
                                {!! Form::select('project_status_filter', $statuses, null, [
                                    'class' => 'form-control select2',
                                    'placeholder' => __('messages.all'),
                                    'style' => 'width: 100%;',
                                ]) !!}
                            </div>
                        </div>
                    @endif
                    <div class="col-md-3">
                        <div class="form-group">
                            {!! Form::label('project_end_date_filter', __('project::lang.end_date') . ':') !!}
                            {!! Form::select('project_end_date_filter', $due_dates, null, [
                                'class' => 'form-control select2',
                                'placeholder' => __('messages.all'),
                                'style' => 'width: 100%;',
                            ]) !!}
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="form-group">
                            {!! Form::label('project_categories_filter', __('project::lang.category') . ':') !!}
                            {!! Form::select('project_categories_filter', $categories, null, [
                                'class' => 'form-controll select2',
                                'placeholder' => __('messages.all'),
                                'style' => 'width:100%;',
                            ]) !!}
                        </div>
                    </div>
                </div>
                @if ($project_view == 'list_view')
                    <div class="project_html">
                    </div>
                @endif
                <!-- project kanban -->
                @if ($project_view == 'kanban')
                    <div class="project-kanban-board">
                        <div class="page">
                            <div class="main">
                                <div class="meta-tasks-wrapper">
                                    <div id="myKanban" class="meta-tasks">
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                @endif
            </div>
        @endcomponent
        <!-- /.box -->
        <div class="modal fade" tabindex="-1" role="dialog" id="project_model"></div>
    </section>
    <link rel="stylesheet" href="{{ asset('modules/project/sass/project.css?v=' . $asset_v) }}">
@endsection
@section('javascript')
    <script src="{{ asset('modules/project/js/project.js?v=' . $asset_v) }}"></script>
    <!-- get list of project on load of page -->
    <script type="text/javascript">
        $(document).ready(function() {
            var project_view = urlSearchParam('project_view');

            //if project view is empty, set default to list_view
            if (_.isEmpty(project_view)) {
                project_view = 'list_view';
            }

            if (project_view == 'kanban') {
                $('.kanban').addClass('active');
                $('.list').removeClass('active');
                initializeProjectKanbanBoard();
            } else if (project_view == 'list_view') {
                getProjectList();
            }
        });
    </script>
@endsection
