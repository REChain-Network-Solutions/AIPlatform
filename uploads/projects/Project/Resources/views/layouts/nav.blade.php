<section class="no-print">
    <nav class="navbar-default tw-transition-all tw-duration-5000 tw-shrink-0 tw-rounded-2xl tw-m-[16px] tw-border-2 !tw-bg-white">
        <div class="container-fluid">
            <!-- Brand and toggle get grouped for better mobile display -->
            <div class="navbar-header">
                <button type="button" class="navbar-toggle collapsed" data-toggle="collapse" data-target="#bs-example-navbar-collapse-1" aria-expanded="false" style="margin-top: 3px; margin-right: 3px;">
                    <span class="sr-only">Toggle navigation</span>
                    <span class="icon-bar"></span>
                    <span class="icon-bar"></span>
                    <span class="icon-bar"></span>
                </button>
                <a class="navbar-brand" href="{{action([\Modules\Project\Http\Controllers\ProjectController::class, 'index']) . '?project_view=list_view'}}"><i class="fas fa-project-diagram"></i> {{__('project::lang.project')}}</a>
            </div>

            <!-- Collect the nav links, forms, and other content for toggling -->
            <div class="collapse navbar-collapse" id="bs-example-navbar-collapse-1">
                <ul class="nav navbar-nav">
                        <li @if(request()->segment(2) == 'project') class="active" @endif><a href="{{action([\Modules\Project\Http\Controllers\ProjectController::class, 'index']) . '?project_view=list_view'}}">@lang('project::lang.projects')</a></li>

                        <li @if(request()->segment(2) == 'project-task') class="active" @endif><a href="{{action([\Modules\Project\Http\Controllers\TaskController::class, 'index'])}}">@lang('project::lang.my_tasks')</a></li>

                    @if($__is_admin)
                        <li @if(request()->segment(2) == 'project-reports') class="active" @endif><a href="{{action([\Modules\Project\Http\Controllers\ReportController::class, 'index'])}}">@lang('report.reports')</a></li>

                        <li @if(request()->get('type') == 'project') class="active" @endif><a href="{{action([\App\Http\Controllers\TaxonomyController::class, 'index']) . '?type=project'}}">@lang('project::lang.project_categories')</a></li>
                    @endif
                    <li @if(request()->segment(2) == 'settings') class="active" @endif><a href="{{action([\Modules\Project\Http\Controllers\SettingsController::class, 'index'])}}"> @lang('project::lang.settings')</a></li>
                </ul>

            </div><!-- /.navbar-collapse -->
        </div><!-- /.container-fluid -->
    </nav>
</section>