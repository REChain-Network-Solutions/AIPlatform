@extends('panel.layout.app', ['disable_tblr' => true])

@section('title', __('New Pipeline — CRM'))
@section('titlebar_title', __('New Pipeline'))
@section('titlebar_subtitle', __('Define stages to track your deals.'))

@section('content')
    <div class="py-10">
        <form method="POST" action="{{ route('dashboard.crm.pipelines.store') }}" id="pipeline-form">
            @csrf
            @include('crm.pipelines._form')

            <div class="mt-6 flex items-center gap-3">
                <x-button type="submit" variant="primary">{{ __('Create Pipeline') }}</x-button>
                <x-button href="{{ route('dashboard.crm.pipelines.index') }}" variant="ghost">{{ __('Cancel') }}</x-button>
            </div>
        </form>
    </div>
@endsection

@push('script')
    @include('crm.pipelines._stage_js')
@endpush
