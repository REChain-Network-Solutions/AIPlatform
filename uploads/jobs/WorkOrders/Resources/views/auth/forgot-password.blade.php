@extends('layouts.auth')
@php
    $settings = settings();
@endphp
@section('tab-title')
    {{ __('Reset Password') }}
@endsection
@push('script-page')
    @if ($settings['google_recaptcha'] == 'on')
        {!! NoCaptcha::renderJs() !!}
    @endif
@endpush
@section('content')

    <div class="card">
        <div class="card-body">
            <div class="row">
                <div class="d-flex justify-content-center">
                    <div class="auth-header">
                        <h2 class="text-secondary"><b>{{ __('Forgot Password ?') }} </b></h2>
                        <p class="f-16 mt-2">{{ __('Enter your email and well send you a link to reset') }}</p>
                    </div>
                </div>
            </div>

            @if (session('error'))
            <div class="alert alert-danger" role="alert">{{ session('error') }}</div>
            @endif
            @if (session('success'))
            <div class="alert alert-success" role="alert">{{ session('success') }}</div>
            @endif
            @if (session('status'))
            <div class="alert alert-success" role="alert">{{ session('status') }}</div>
            @endif

            {{ Form::open(['route' => 'password.email', 'method' => 'post', 'id' => 'forgotpasswordForm']) }}
            <div class="form-floating mb-3">
                <input type="email" class="form-control" id="email" name="email"
                    placeholder="{{ __('Email address') }}" />
                <label for="email">{{ __('Email address') }}</label>
                @error('email')
                    <span class="invalid-email text-danger" role="alert">
                        <strong>{{ $message }}</strong>
                    </span>
                @enderror
            </div>

        @if ($settings['google_recaptcha'] == 'on')
                <div class="form-group">
                    <label for="email" class="form-label"></label>
                    {!! NoCaptcha::display() !!}
                    @error('g-recaptcha-response')
                        <span class="small text-danger" role="alert">
                            <strong>{{ $message }}</strong>
                        </span>
                    @enderror
                </div>
            @endif

            <div class="d-grid mt-4">
                <button type="submit" class="btn btn-secondary p-2">{{ __('Send Reset Link') }}</button>
            </div>
            <hr />
            <h5 class="d-flex justify-content-center">{{ __('Back to') }} <a class="ms-1 text-secondary"
                    href="{{ route('login') }}">{{ __('Log In') }}</a>
            </h5>
            {{ Form::close() }}
        </div>
    </div>
@endsection
