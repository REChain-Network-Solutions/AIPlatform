@extends('panel.layout.app', ['disable_tblr' => true])
@section('title', __('New Deal — CRM'))
@section('titlebar_title', __('New Deal'))
@section('titlebar_back', route('dashboard.crm.deals.index'))

@section('content')
    <div class="mx-auto max-w-2xl py-10">
        <form method="POST" action="{{ route('dashboard.crm.deals.store') }}" class="space-y-6">
            @csrf
            @include('crm.deals._form', ['deal' => null])
            <div class="flex items-center gap-3">
                <x-button type="submit" variant="primary">{{ __('Save Deal') }}</x-button>
                <a href="{{ route('dashboard.crm.deals.index') }}" class="text-sm text-muted-foreground hover:underline">{{ __('Cancel') }}</a>
            </div>
        </form>
    </div>
@endsection
