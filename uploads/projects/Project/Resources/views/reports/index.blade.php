@extends('layouts.app')
@section('title', __('project::lang.project_report'))

@section('content')
@include('project::layouts.nav')
<section class="content-header">
	<h1 class="tw-text-xl md:tw-text-3xl tw-font-bold tw-text-black">
    	@lang('project::lang.project_report')
    </h1>
</section>
<section class="content">
	<div class="row">
		<div class="col-md-4">
			@component('components.widget')
			<div class="box-body text-center">
				<i class="fas fa-hourglass-half fs-20"></i> <br>
				<span class="fs-20">
					@lang('project::lang.time_log_report') <br>
					<small>@lang('project::lang.by_employees')</small>
				</span>
			</div>
			<div class="box-footer text-center">
				<a href="{{action([\Modules\Project\Http\Controllers\ReportController::class, 'getEmployeeTimeLogReport'])}}" class="tw-dw-btn tw-dw-btn-neutral tw-text-white tw-dw-btn-sm tw-dw-btn-wide">
					<i class="fa fa-eye"></i>
					@lang("messages.view")
				</a>
			</div>
			@endcomponent
		</div>
		<div class="col-md-4">
			@component('components.widget')
			<div class="box-body text-center">
				<i class="fas fa-hourglass-half fs-20"></i> <br>
				<span class="fs-20">
					@lang('project::lang.time_log_report') <br>
					<small>@lang('project::lang.by_projects')</small>
				</span>
			</div>
			<div class="box-footer text-center">
				<a href="{{action([\Modules\Project\Http\Controllers\ReportController::class, 'getProjectTimeLogReport'])}}" class="tw-dw-btn tw-dw-btn-neutral tw-text-white tw-dw-btn-sm tw-dw-btn-wide">
					<i class="fa fa-eye"></i>
					@lang("messages.view")
				</a>
			</div>
			@endcomponent
		</div>
	</div>
<link rel="stylesheet" href="{{ asset('modules/project/sass/project.css?v=' . $asset_v) }}">
</section>
@endsection