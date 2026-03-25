@extends('layouts.auth')
@section('tab-title')
    {{ __('Reset Password') }}
@endsection

@section('content')
    <div class="card">
        <div class="card-body">
            <div class="row">
                <div class="d-flex justify-content-center">
                    <div class="auth-header">
                        <h2 class="text-secondary"><b>{{ __('Reset your password') }} </b></h2>
                        <p class="f-16 mt-2">{{ __('You have successfully verified your account. Enter') }} <br>
                            {{ __('new password below.') }}</p>
                    </div>
                </div>
            </div>



            {{ Form::open(['route' => 'password.update', 'method' => 'post', 'id' => 'loginForm']) }}
            @if (session('error'))
                <div class="alert alert-danger" role="alert">{{ session('error') }}</div>
            @endif
            @if (session('success'))
                <div class="alert alert-success" role="alert">{{ session('success') }}</div>
            @endif
            @if (session('status'))
                <div class="alert alert-success" role="alert">{{ session('status') }}</div>
            @endif
            <input type="hidden" name="token" value="{{ $request->route('token') }}">
            <div class="form-floating mb-3">
                <input type="email" class="form-control" id="email" name="email"
                    value="{{ $request->get('email') }}" placeholder="{{ __('Email address') }}" />
                <label for="email">{{ __('Email address') }}</label>
                @error('email')
                    <span class="invalid-email text-danger" role="alert">
                        <strong>{{ $message }}</strong>
                    </span>
                @enderror
            </div>
            <div class="form-floating mb-3">
                <input type="password" class="form-control" id="password" name="password"
                    placeholder="{{ __('Password') }}" />
                <label for="password">{{ __('Password') }}</label>
                @error('password')
                    <span class="invalid-password text-danger" role="alert">
                        <strong>{{ $message }}</strong>
                    </span>
                @enderror
            </div>

            <div class="form-floating mb-3">
                <input type="password" class="form-control" id="password_confirmation" name="password_confirmation"
                    placeholder="{{ __('Password Confirmation') }}" />
                <label for="password_confirmation">{{ __('Password Confirmation') }}</label>
                @error('password_confirmation')
                    <span class="invalid-password_confirmation text-danger" role="alert">
                        <strong>{{ $message }}</strong>
                    </span>
                @enderror
            </div>
            <div class="d-grid mt-4">
                <button type="submit" class="btn btn-secondary p-2">{{ __('Update Password') }}</button>
            </div>

            {{ Form::close() }}
        </div>
    </div>
@endsection that dd not show in this page
