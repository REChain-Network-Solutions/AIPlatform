@extends('panel.layout.app', ['disable_tblr' => true])
@section('title', __('Edit Deal — CRM'))
@section('titlebar_title', __('Edit Deal'))
@section('titlebar_back', route('dashboard.crm.deals.show', $deal))

@section('content')
    <div class="mx-auto max-w-2xl py-10">
        <form method="POST" action="{{ route('dashboard.crm.deals.update', $deal) }}" class="space-y-6">
            @csrf
            @method('PUT')
            @include('crm.deals._form', ['deal' => $deal])
            <div class="flex items-center gap-3">
                <x-button type="submit" variant="primary">{{ __('Update Deal') }}</x-button>
                <a href="{{ route('dashboard.crm.deals.show', $deal) }}" class="text-sm text-muted-foreground hover:underline">{{ __('Cancel') }}</a>
            </div>
        </form>
    </div>
@endsection
