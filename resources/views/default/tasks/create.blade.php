@extends('panel.layout.app', ['disable_tblr' => true])

@section('title', __('New Task'))
@section('titlebar_title', __('New Task'))
@section('titlebar_subtitle', isset($parentTask) ? __('Sub-task of: ') . $parentTask->title : __('Create a new task.'))

@section('content')
    <div class="py-10">
        <form method="POST" action="{{ route('dashboard.tasks.store') }}" id="task-form">
            @csrf
            @include('tasks._form')
            <div class="mt-6 flex items-center gap-3">
                <x-button type="submit" variant="primary">{{ __('Create Task') }}</x-button>
                <x-button href="{{ route('dashboard.tasks.index') }}" variant="ghost">{{ __('Cancel') }}</x-button>
            </div>
        </form>
    </div>
@endsection
