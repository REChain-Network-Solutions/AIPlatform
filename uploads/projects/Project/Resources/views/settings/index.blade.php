@extends('layouts.app')
@section('title', __('project::lang.settings'))

@section('content')
@include('project::layouts.nav')
<section class="content-header">
	<h1 class="tw-text-xl md:tw-text-3xl tw-font-bold tw-text-black" >
    	@lang('project::lang.settings')
    </h1>
</section>
<section class="content">
            @component('components.widget')
            <div class="box-body">
                {!! Form::open([
                    'url' => action([\Modules\Project\Http\Controllers\SettingsController::class, 'store']),
                    'method' => 'post',
                    'id' => 'settings',
                    'files' => true
                ]) !!}

                @php
                $fields = json_decode($busines->prj_setting) ?? [];
                @endphp
                <div class="row">
                    <div class="col-md-4">
                        <div class="form-group">
                            {!! Form::label('custom_field1', __('project::lang.custom_field1')) !!}
                            {!! Form::text('custom_fields[custom_field1]', $fields->custom_fields->custom_field1 ?? null , [
                                'class' => 'form-control',
                                'placeholder' => __('project::lang.custom_field1'),
                            ]) !!}
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="form-group">
                            {!! Form::label('custom_field2', __('project::lang.custom_field2')) !!}
                            {!! Form::text('custom_fields[custom_field2]', $fields->custom_fields->custom_field2 ?? null, [
                                'class' => 'form-control',
                                'placeholder' => __('project::lang.custom_field2'),
                            ]) !!}
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="form-group">
                            {!! Form::label('custom_field3', __('project::lang.custom_field3')) !!}
                            {!! Form::text('custom_fields[custom_field3]', $fields->custom_fields->custom_field3 ?? null, [
                                'class' => 'form-control',
                                'placeholder' => __('project::lang.custom_field3'),
                            ]) !!}
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="form-group">
                            {!! Form::label('custom_field4', __('project::lang.custom_field4')) !!}
                            {!! Form::text('custom_fields[custom_field4]', $fields->custom_fields->custom_field4 ?? null, [
                                'class' => 'form-control',
                                'placeholder' => __('project::lang.custom_field4'),
                            ]) !!}
                        </div>
                    </div>
                </div>
                <div class="col-md-12 text-center">
                    <button type="submit" name="submit_action" value="save" class="tw-dw-btn tw-dw-btn-primary tw-dw-btn-lg tw-text-white submit_form">@lang('messages.save')</button>
                </div>
                {!! Form::close() !!}
            </div>
            @endcomponent
    </section>
@endsection