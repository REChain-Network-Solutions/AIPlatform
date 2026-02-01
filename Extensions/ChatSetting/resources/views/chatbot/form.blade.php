@extends('panel.layout.app')
@section('title', $title)
@section('titlebar_actions', '')
@section('content')
    <div class="page-body pt-6">
        <div class="container-xl">
            <div class="row">
                <div class="col-md-6 mx-auto">
                    <form class="@if(view()->exists('panel.admin.custom.layout.panel.header-top-bar')) bg-[--tblr-bg-surface] px-8 py-10 rounded-[--tblr-border-radius] @endif" id="user_edit_form" method="post" enctype="multipart/form-data" action="{{ $action }}">
                        @csrf
                        @method($method)
                        <input type="hidden" name="model" value="{{ \App\Helpers\Classes\Helper::setting('openai_default_model') }}">
                        <div class="row">
                            <div class="col-md-12 col-xl-12">
                                <div class="row">
                                    <div class="col-md-12">
                                        <p>You have the ability to provide directives to your personalized GPT and
                                            tailor it according to your preferences, ensuring it aligns seamlessly with
                                            your brand and tone.</p>
                                        <div class="mb-4 rounded-xl  p-2"
                                             style="background-color: rgba(157, 107, 221, 0.1);">
                                            <p class="fs-3 font-bold font-weight-medium mb-0 pb-0">
                                                <span class="ps-2 pe-2 rounded me-2 text-white" style="background-color: rgba(157, 107, 221, 1)">1</span>@lang('General')
                                            </p>
                                        </div>
                                    </div>
                                    <div class="col-md-12">
                                        <div class="mb-3">
                                            <label for="title" class="form-label">{{__('Title')}}</label>
                                            <input type="text" class="form-control @error('title') is-invalid @enderror"
                                                   id="title" name="title" value="{{ old('title', $item->title) }}">
                                            @error('title')
                                            <div class="invalid-feedback">
                                                {{ $message }}
                                            </div>
                                            @enderror
                                        </div>
                                    </div>
                                    <div class="col-md-12">
                                        <div class="mb-3">
                                            <label for="role" class="form-label">{{__('Role')}}</label>
                                            <input type="text" class="form-control @error('role') is-invalid @enderror"
                                                   id="role" name="role" value="{{ old('role', $item->role) }}">
                                            @error('role')
                                            <div class="invalid-feedback">
                                                {{ $message }}
                                            </div>
                                            @enderror
                                        </div>
                                    </div>
                                    <div class="col-md-12">
                                        <div class="mb-3">
                                            <label for="first_message"
                                                   class="form-label">{{__('First Message')}}</label>
                                            <input type="text"
                                                   class="form-control @error('first_message') is-invalid @enderror"
                                                   id="first_message" name="first_message"
                                                   value="{{ old('first_message', $item->first_message) }}">
                                            @error('first_message')
                                            <div class="invalid-feedback">
                                                {{ $message }}
                                            </div>
                                            @enderror
                                        </div>
                                    </div>
                                    <div class="col-md-12">
                                        <div class="mb-3">
                                            <label for="instructions" class="form-label">
                                                {{__('Instructions')}}
                                                <x-info-tooltip :text="__('You can provide instructions to your GPT-3 model to ensure it aligns with your brand and tone.')"/>
                                            </label>
                                            <textarea type="text" class="form-control @error('instructions') is-invalid @enderror" id="instructions" name="instructions">{{ old('instructions', $item->instructions) }}</textarea>
                                            @error('instructions')
                                            <div class="invalid-feedback">
                                                {{ $message }}
                                            </div>
                                            @enderror
                                        </div>
                                    </div>
                                    {{--                                    <div class="col-md-12">--}}
                                    {{--                                        <div class="mb-4 rounded-xl  p-2"--}}
                                    {{--                                             style="background-color: rgba(157, 107, 221, 0.1);">--}}
                                    {{--                                            <p class="fs-3 font-bold font-weight-medium mb-0 pb-0"><span--}}
                                    {{--                                                        class="ps-2 pe-2 rounded me-2 text-white"--}}
                                    {{--                                                        style="background-color: rgba(157, 107, 221, 1)">2</span>@lang('Customization')--}}
                                    {{--                                            </p>--}}
                                    {{--                                        </div>--}}
                                    {{--                                    </div>--}}
                                </div>

                                {{--                                <div class="row">--}}
                                {{--                                    <div class="col-12">--}}
                                {{--                                        <div class="mb-3">--}}
                                {{--                                            <label class="form-label">@lang('Logo')</label>--}}
                                {{--                                            <input type="file" class="form-control @error('logo') is-invalid @enderror" id="logo" name="logo">--}}
                                {{--											@error('logo')--}}
                                {{--											<div class="invalid-feedback">--}}
                                {{--												{{ $message }}--}}
                                {{--											</div>--}}
                                {{--											@enderror--}}
                                {{--                                        </div>--}}
                                {{--                                    </div>--}}
                                {{--                                    <div class="col-md-12">--}}
                                {{--                                        <div class="mb-3">--}}
                                {{--                                            <label for="width" class="form-label">{{__('Chat Panel Width')}}</label>--}}
                                {{--                                            <input type="number" class="form-control @error('width') is-invalid @enderror" id="width" name="width"--}}
                                {{--                                                   value="{{ old('width', $item->width) }}">--}}
                                {{--											@error('width')--}}
                                {{--											<div class="invalid-feedback">--}}
                                {{--												{{ $message }}--}}
                                {{--											</div>--}}
                                {{--											@enderror--}}
                                {{--                                        </div>--}}
                                {{--                                    </div>--}}
                                {{--                                    <div class="col-md-12">--}}
                                {{--                                        <div class="mb-3">--}}
                                {{--                                            <label for="height" class="form-label">{{__('Chat Panel Height')}}</label>--}}
                                {{--                                            <input type="number" class="form-control @error('height') is-invalid @enderror" id="height" name="height"--}}
                                {{--                                                   value="{{ old('height', $item->height) }}">--}}
                                {{--											@error('height')--}}
                                {{--											<div class="invalid-feedback">--}}
                                {{--												{{ $message }}--}}
                                {{--											</div>--}}
                                {{--											@enderror--}}
                                {{--                                        </div>--}}
                                {{--                                    </div>--}}
                                {{--                                    <div class="col-md-12">--}}
                                {{--                                        <div class="mb-3">--}}
                                {{--                                            <label for="color" class="form-label">{{__('Color')}}</label>--}}
                                {{--                                            <input type="color" class="form-control @error('color') is-invalid @enderror" id="color" name="color"--}}
                                {{--                                                   value="{{ old('color', $item->color) }}">--}}
                                {{--											@error('color')--}}
                                {{--											<div class="invalid-feedback">--}}
                                {{--												{{ $message }}--}}
                                {{--											</div>--}}
                                {{--											@enderror--}}
                                {{--                                        </div>--}}
                                {{--                                    </div>--}}
                                {{--                                </div>--}}
                                <button form="user_edit_form" id="user_edit_button" class="btn btn-primary w-100">
                                    {{__('Save')}}
                                </button>
                            </div>
                        </div>
                    </form>
                </div>
            </div>
        </div>
    </div>
@endsection
@section('script')
@endsection
