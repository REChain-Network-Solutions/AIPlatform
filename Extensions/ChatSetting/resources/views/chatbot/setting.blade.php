@extends('panel.layout.app')
@section('title', $title)

@section('content')
    <div class="page-header">
        <div class="container-xl">
            <div class="row g-2 items-center">
                <div class="col">
                    <div class="hstack gap-1">
                        <a href="{{ LaravelLocalization::localizeUrl( route('dashboard.index') ) }}"
                           class="page-pretitle flex items-center">
                            <svg class="!me-2 rtl:-scale-x-100" width="8" height="10" viewBox="0 0 6 10"
                                 fill="currentColor" xmlns="http://www.w3.org/2000/svg">
                                <path d="M4.45536 9.45539C4.52679 9.45539 4.60714 9.41968 4.66071 9.36611L5.10714 8.91968C5.16071 8.86611 5.19643 8.78575 5.19643 8.71432C5.19643 8.64289 5.16071 8.56254 5.10714 8.50896L1.59821 5.00004L5.10714 1.49111C5.16071 1.43753 5.19643 1.35718 5.19643 1.28575C5.19643 1.20539 5.16071 1.13396 5.10714 1.08039L4.66071 0.633963C4.60714 0.580392 4.52679 0.544678 4.45536 0.544678C4.38393 0.544678 4.30357 0.580392 4.25 0.633963L0.0892856 4.79468C0.0357141 4.84825 0 4.92861 0 5.00004C0 5.07146 0.0357141 5.15182 0.0892856 5.20539L4.25 9.36611C4.30357 9.41968 4.38393 9.45539 4.45536 9.45539Z"/>
                            </svg>
                            {{__('Back to dashboard')}}
                        </a>
                        <a href="{{route('dashboard.admin.chatbot.index')}}" class="page-pretitle flex items-center">
                            / {{__('Chatbot Templates')}}
                        </a>
                    </div>
                    <h2 class="page-title mb-2">
                        {{ $title }}
                    </h2>
                </div>
            </div>
        </div>
    </div>
    <!-- Page body -->
    <div class="page-body pt-6">
        <div class="container-xl">
            <div class="row">
                <div class="col-md-6 mx-auto">
                    <form class="@if(view()->exists('panel.admin.custom.layout.panel.header-top-bar')) bg-[--tblr-bg-surface] px-8 py-10 rounded-[--tblr-border-radius] @endif"
                          id="user_edit_form" method="post" enctype="multipart/form-data" action="{{ $action }}">
                        @csrf
                        @method($method)
                        <div class="row">
                            <div class="col-md-12">
                                <div class="mb-3">
                                    <label class="form-label" for="status">{{__('Show ChatBot Ballon on')}}</label>
                                    <select class="form-select @error('chatbot_status') is-invalid @enderror" name="chatbot_status" id="chatbot_status">
                                        <option value="disabled" {{$settings_two->chatbot_status == 'disabled' ? 'selected' : null}}>{{__('Disabled')}}</option>
                                        <option value="frontend" {{$settings_two->chatbot_status == 'frontend' ? 'selected' : null}}>{{__('Frontend')}}</option>
                                        <option value="dashboard" {{$settings_two->chatbot_status == 'dashboard' ? 'selected' : null}}>{{__('Dashboard')}}</option>
                                        <option value="both" {{$settings_two->chatbot_status == 'both' ? 'selected' : null}}>{{__('Both')}}</option>
                                    </select>
                                    @error('chatbot_status')
                                    <div class="invalid-feedback">
                                        {{$message}}
                                    </div>
                                    @enderror
                                </div>
                                <div class="mb-3">
                                    <label class="form-label" for="template">{{__('Template')}}</label>
                                    <select class="form-select @error('chatbot_template') is-invalid @enderror" name="chatbot_template" id="chatbot_template">
                                        @foreach($chatbotData as $chatbot)
                                            <option value="{{$chatbot->id}}" {{$settings_two->chatbot_template == $chatbot->id ? 'selected' : null}}>{{__($chatbot->title)}}</option>
                                        @endforeach
                                    </select>
                                    @error('chatbot_template')
                                    <div class="invalid-feedback">
                                        {{$message}}
                                    </div>
                                    @enderror
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">{{__('Position')}}</label>
                                    <div class="form-selectgroup">
                                        <label class="form-selectgroup-item">
                                            <input type="radio" name="position" value="top-left" class="form-selectgroup-input" {{$settings_two->chatbot_position == 'top-left' ? 'checked' : null}} />
                                            <span class="form-selectgroup-label">
                                                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" fill="none" stroke-linecap="round" stroke-linejoin="round">
                                                    <path stroke="none" d="M0 0h24v24H0z" fill="none"/>
                                                    <path d="M10 3h-5a2 2 0 0 0 -2 2v5a2 2 0 0 0 2 2h5a2 2 0 0 0 2 -2v-5a2 2 0 0 0 -2 -2z" stroke-width="0" fill="currentColor" />
                                                    <path d="M15 3a1 1 0 0 1 .117 1.993l-.127 .007a1 1 0 0 1 -.117 -1.993l.127 -.007z" stroke-width="0" fill="currentColor" />
                                                    <path d="M20 3a1 1 0 0 1 .117 1.993l-.127 .007a1 1 0 0 1 -.117 -1.993l.127 -.007z" stroke-width="0" fill="currentColor" />
                                                    <path d="M20 8a1 1 0 0 1 .117 1.993l-.127 .007a1 1 0 0 1 -.117 -1.993l.127 -.007z" stroke-width="0" fill="currentColor" />
                                                    <path d="M20 14a1 1 0 0 1 .117 1.993l-.127 .007a1 1 0 0 1 -.117 -1.993l.127 -.007z" stroke-width="0" fill="currentColor" />
                                                    <path d="M4 14a1 1 0 0 1 .117 1.993l-.127 .007a1 1 0 0 1 -.117 -1.993l.127 -.007z" stroke-width="0" fill="currentColor" />
                                                    <path d="M20 19a1 1 0 0 1 .117 1.993l-.127 .007a1 1 0 0 1 -.117 -1.993l.127 -.007z" stroke-width="0" fill="currentColor" />
                                                    <path d="M15 19a1 1 0 0 1 .117 1.993l-.127 .007a1 1 0 0 1 -.117 -1.993l.127 -.007z" stroke-width="0" fill="currentColor" />
                                                    <path d="M9 19a1 1 0 0 1 .117 1.993l-.127 .007a1 1 0 0 1 -.117 -1.993l.127 -.007z" stroke-width="0" fill="currentColor" />
                                                    <path d="M4 19a1 1 0 0 1 .117 1.993l-.127 .007a1 1 0 0 1 -.117 -1.993l.127 -.007z" stroke-width="0" fill="currentColor" />
                                                </svg>
                                                {{-- {{__('Top Left')}} --}}
                                            </span>
                                        </label>
                                        <label class="form-selectgroup-item">
                                            <input type="radio" name="position" value="top-right" class="form-selectgroup-input" {{$settings_two->chatbot_position == 'top-right' ? 'checked' : null}} />
                                            <span class="form-selectgroup-label">
                                                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" fill="none" stroke-linecap="round" stroke-linejoin="round">
                                                    <path stroke="none" d="M0 0h24v24H0z" fill="none"/>
                                                    <path d="M19 3.01h-5a2 2 0 0 0 -2 2v5a2 2 0 0 0 2 2h5a2 2 0 0 0 2 -2v-5a2 2 0 0 0 -2 -2z" stroke-width="0" fill="currentColor" />
                                                    <path d="M20 14a1 1 0 0 1 .993 .883l.007 .127a1 1 0 0 1 -1.993 .117l-.007 -.127a1 1 0 0 1 1 -1z" stroke-width="0" fill="currentColor" />
                                                    <path d="M20 19a1 1 0 0 1 .993 .883l.007 .127a1 1 0 0 1 -1.993 .117l-.007 -.127a1 1 0 0 1 1 -1z" stroke-width="0" fill="currentColor" />
                                                    <path d="M15 19a1 1 0 0 1 .993 .883l.007 .127a1 1 0 0 1 -1.993 .117l-.007 -.127a1 1 0 0 1 1 -1z" stroke-width="0" fill="currentColor" />
                                                    <path d="M9 19a1 1 0 0 1 .993 .883l.007 .127a1 1 0 0 1 -1.993 .117l-.007 -.127a1 1 0 0 1 1 -1z" stroke-width="0" fill="currentColor" />
                                                    <path d="M9 3a1 1 0 0 1 .993 .883l.007 .127a1 1 0 0 1 -1.993 .117l-.007 -.127a1 1 0 0 1 1 -1z" stroke-width="0" fill="currentColor" />
                                                    <path d="M4 19a1 1 0 0 1 .993 .883l.007 .127a1 1 0 0 1 -1.993 .117l-.007 -.127a1 1 0 0 1 1 -1z" stroke-width="0" fill="currentColor" />
                                                    <path d="M4 14a1 1 0 0 1 .993 .883l.007 .127a1 1 0 0 1 -1.993 .117l-.007 -.127a1 1 0 0 1 1 -1z" stroke-width="0" fill="currentColor" />
                                                    <path d="M4 8a1 1 0 0 1 .993 .883l.007 .127a1 1 0 0 1 -1.993 .117l-.007 -.127a1 1 0 0 1 1 -1z" stroke-width="0" fill="currentColor" />
                                                    <path d="M4 3a1 1 0 0 1 .993 .883l.007 .127a1 1 0 0 1 -1.993 .117l-.007 -.127a1 1 0 0 1 1 -1z" stroke-width="0" fill="currentColor" />
                                                </svg>
                                                {{-- {{__('Top Right')}} --}}
                                            </span>
                                        </label>
                                        <label class="form-selectgroup-item">
                                            <input type="radio" name="position" value="bottom-right" class="form-selectgroup-input" {{$settings_two->chatbot_position == 'bottom-right' ? 'checked' : null}} />
                                            <span class="form-selectgroup-label">
                                                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" fill="none" stroke-linecap="round" stroke-linejoin="round">
                                                    <path stroke="none" d="M0 0h24v24H0z" fill="none"/>
                                                    <path d="M19 12h-5a2 2 0 0 0 -2 2v5a2 2 0 0 0 2 2h5a2 2 0 0 0 2 -2v-5a2 2 0 0 0 -2 -2z" stroke-width="0" fill="currentColor" />
                                                    <path d="M20 8a1 1 0 0 1 .993 .883l.007 .127a1 1 0 0 1 -1.993 .117l-.007 -.127a1 1 0 0 1 1 -1z" stroke-width="0" fill="currentColor" />
                                                    <path d="M20 3a1 1 0 0 1 .993 .883l.007 .127a1 1 0 0 1 -1.993 .117l-.007 -.127a1 1 0 0 1 1 -1z" stroke-width="0" fill="currentColor" />
                                                    <path d="M15 3a1 1 0 0 1 .993 .883l.007 .127a1 1 0 0 1 -1.993 .117l-.007 -.127a1 1 0 0 1 1 -1z" stroke-width="0" fill="currentColor" />
                                                    <path d="M9 3a1 1 0 0 1 .993 .883l.007 .127a1 1 0 0 1 -1.993 .117l-.007 -.127a1 1 0 0 1 1 -1z" stroke-width="0" fill="currentColor" />
                                                    <path d="M9 19a1 1 0 0 1 .993 .883l.007 .127a1 1 0 0 1 -1.993 .117l-.007 -.127a1 1 0 0 1 1 -1z" stroke-width="0" fill="currentColor" />
                                                    <path d="M4 3a1 1 0 0 1 .993 .883l.007 .127a1 1 0 0 1 -1.993 .117l-.007 -.127a1 1 0 0 1 1 -1z" stroke-width="0" fill="currentColor" />
                                                    <path d="M4 8a1 1 0 0 1 .993 .883l.007 .127a1 1 0 0 1 -1.993 .117l-.007 -.127a1 1 0 0 1 1 -1z" stroke-width="0" fill="currentColor" />
                                                    <path d="M4 14a1 1 0 0 1 .993 .883l.007 .127a1 1 0 0 1 -1.993 .117l-.007 -.127a1 1 0 0 1 1 -1z" stroke-width="0" fill="currentColor" />
                                                    <path d="M4 19a1 1 0 0 1 .993 .883l.007 .127a1 1 0 0 1 -1.993 .117l-.007 -.127a1 1 0 0 1 1 -1z" stroke-width="0" fill="currentColor" />
                                                </svg>
                                                {{-- {{__('Bottom Right')}} --}}
                                            </span>
                                        </label>
                                        <label class="form-selectgroup-item">
                                            <input type="radio" name="position" value="bottom-left" class="form-selectgroup-input" {{$settings_two->chatbot_position == 'bottom-left' ? 'checked' : null}} />
                                            <span class="form-selectgroup-label">
                                                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" fill="none" stroke-linecap="round" stroke-linejoin="round">
                                                    <path stroke="none" d="M0 0h24v24H0z" fill="none"/>
                                                    <path d="M10 12h-5a2 2 0 0 0 -2 2v5a2 2 0 0 0 2 2h5a2 2 0 0 0 2 -2v-5a2 2 0 0 0 -2 -2z" stroke-width="0" fill="currentColor" />
                                                    <path d="M4 8a1 1 0 0 1 .993 .883l.007 .127a1 1 0 0 1 -1.993 .117l-.007 -.127a1 1 0 0 1 1 -1z" stroke-width="0" fill="currentColor" />
                                                    <path d="M4 3a1 1 0 0 1 .993 .883l.007 .127a1 1 0 0 1 -1.993 .117l-.007 -.127a1 1 0 0 1 1 -1z" stroke-width="0" fill="currentColor" />
                                                    <path d="M9 3a1 1 0 0 1 .993 .883l.007 .127a1 1 0 0 1 -1.993 .117l-.007 -.127a1 1 0 0 1 1 -1z" stroke-width="0" fill="currentColor" />
                                                    <path d="M15 3a1 1 0 0 1 .993 .883l.007 .127a1 1 0 0 1 -1.993 .117l-.007 -.127a1 1 0 0 1 1 -1z" stroke-width="0" fill="currentColor" />
                                                    <path d="M15 19a1 1 0 0 1 .993 .883l.007 .127a1 1 0 0 1 -1.993 .117l-.007 -.127a1 1 0 0 1 1 -1z" stroke-width="0" fill="currentColor" />
                                                    <path d="M20 3a1 1 0 0 1 .993 .883l.007 .127a1 1 0 0 1 -1.993 .117l-.007 -.127a1 1 0 0 1 1 -1z" stroke-width="0" fill="currentColor" />
                                                    <path d="M20 8a1 1 0 0 1 .993 .883l.007 .127a1 1 0 0 1 -1.993 .117l-.007 -.127a1 1 0 0 1 1 -1z" stroke-width="0" fill="currentColor" />
                                                    <path d="M20 14a1 1 0 0 1 .993 .883l.007 .127a1 1 0 0 1 -1.993 .117l-.007 -.127a1 1 0 0 1 1 -1z" stroke-width="0" fill="currentColor" />
                                                    <path d="M20 19a1 1 0 0 1 .993 .883l.007 .127a1 1 0 0 1 -1.993 .117l-.007 -.127a1 1 0 0 1 1 -1z" stroke-width="0" fill="currentColor" />
                                                </svg>
                                                {{-- {{__('Bottom Left')}} --}}
                                            </span>
                                        </label>
                                    </div>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label" for="rate_limit">{{__('Message Limit per day')}}</label>
                                    <input class="form-control @error('chatbot_rate_limit') is-invalid @enderror" type="number" name="chatbot_rate_limit" id="chatbot_rate_limit" min="2" max="50" default="2" value="{{$settings_two->chatbot_rate_limit}}">
                                    @error('chatbot_rate_limit')
                                    <div class="invalid-feedback">
                                        {{$message}}
                                    </div>
                                    @enderror
                                </div>
                                <div class="mb-3">
                                    <label class="form-check form-switch" for="logged_in">
                                        <input class="form-check-input" name="chatbot_login_require" type="checkbox" id="chatbot_login_require" {{ $settings_two->chatbot_login_require == true ? 'checked' : '' }}>
                                        <span class="form-check-label">{{ __('Disable if user not logged in?') }}</span>
                                    </label>
                                </div>
                            </div>
                        </div>
                        <button type="submit" class="btn btn-primary w-100">
                            {{__('Save')}}
                        </button>
                    </form>
                </div>
            </div>
        </div>
    </div>
@endsection
@section('script')
@endsection
