@extends('layouts.app')

@section('content')
    <div class="content-wrapper">
        <!-- EDIT TASK START -->
        <div class="bg-white rounded b-shadow-4 create-inv">
            <!-- HEADING START -->
            <div class="px-lg-4 px-md-4 px-3 py-3">
                <h4 class="mb-0 f-21 font-weight-normal">@lang('app.menu.tasksAssignment')</h4>
            </div>
            <!-- HEADING END -->
            <hr class="m-0 border-top-grey">
            <!-- FORM START -->
            <x-form class="c-inv-form" id="saveTaskForm" method="PUT">
                <!-- DATE, TASK, ASSIGNEE START -->
                <div class="row px-lg-4 px-md-4 px-3 py-3">
                    <!-- DATE START -->
                    <div class="col-md-6">
                        <div class="form-group mb-lg-0 mb-md-0 mb-4">
                            <x-forms.label fieldId="date" :fieldLabel="__('Date')" fieldRequired="true"></x-forms.label>
                            <div class="input-group">
                                <input type="text" id="date" name="date"
                                    class="px-6 position-relative text-dark font-weight-normal form-control height-35 rounded p-0 text-left f-15"
                                    placeholder="@lang('placeholders.date')"
                                    value="{{ $task->date ? $task->date->format(company()->date_format) : '' }}">
                            </div>
                        </div>
                    </div>
                    <!-- DATE END -->

                    <!-- PRIORITY START -->
                    <div class="col-md-6">
                        <div class="form-group c-inv-select mb-lg-0 mb-md-0 mb-4">
                            <x-forms.label fieldId="priority" :fieldLabel="__('Priority')" fieldRequired="true"></x-forms.label>
                            <div class="select-others height-35 rounded">
                                <select class="form-control select-picker" name="priority" id="priority">
                                    <option value="High" @selected($task->priority == 'High')>@lang('High')</option>
                                    <option value="Medium" @selected($task->priority == 'Medium')>@lang('Medium')</option>
                                    <option value="Low" @selected($task->priority == 'Low')>@lang('Low')</option>
                                </select>
                            </div>
                        </div>
                    </div>
                    <!-- PRIORITY END -->
                </div>

                <div class="row px-lg-4 px-md-4 px-3 py-3">
                    <!-- TASK START -->
                    <div class="col-md-12">
                        <div class="form-group mb-4">
                            <x-forms.label fieldId="task" :fieldLabel="__('Task')" fieldRequired="true"></x-forms.label>
                            <textarea class="form-control f-14 pt-2" rows="3" name="task" id="task"
                                placeholder="@lang('placeholders.taskDescription')">{{ $task->task }}</textarea>
                        </div>
                    </div>
                    <!-- TASK END -->
                </div>

                <div class="row px-lg-4 px-md-4 px-3 py-3">
                    <!-- ASSIGNEE START -->
                    <div class="col-md-6">
                        <div class="form-group mb-4">
                            <x-forms.label fieldId="assign_to" :fieldLabel="__('Assigned To')" fieldRequired="true"></x-forms.label>
                            <input type="text" class="form-control height-35 f-15" name="assign_to" id="assign_to"
                                value="{{ $task->assign_to }}">
                        </div>
                    </div>
                    <!-- ASSIGNEE END -->

                    <!-- PRODUCT START -->
                    <div class="col-md-6">
                        <div class="form-group mb-4">
                            <x-forms.label fieldId="product" :fieldLabel="__('Product')" fieldRequired="true"></x-forms.label>
                            <input type="text" class="form-control height-35 f-15" name="product" id="product"
                                value="{{ $task->product }}">
                        </div>
                    </div>
                    <!-- PRODUCT END -->
                </div>
                <!-- DATE, TASK, ASSIGNEE END -->

                <div class="row px-lg-4 px-md-4 px-3 py-3">
                    <!-- STATUS START -->
                    <div class="col-md-6">
                        <div class="form-group c-inv-select mb-lg-0 mb-md-0 mb-4">
                            <x-forms.label fieldId="status" :fieldLabel="__('Status')" fieldRequired="true"></x-forms.label>
                            <div class="select-others height-35 rounded">
                                <select class="form-control select-picker" name="status" id="status">
                                    <option value="Not Yet Started" @selected($task->status == 'Not Yet Started')>@lang('Not Yet Started')</option>
                                    <option value="In Progress" @selected($task->status == 'In Progress')>@lang('In Progress')</option>
                                    <option value="Completed" @selected($task->status == 'Completed')>@lang('Completed')</option>
                                </select>
                            </div>
                        </div>
                    </div>
                    <!-- STATUS END -->

                    <!-- ETA START -->
                    <div class="col-md-6">
                        <div class="form-group mb-4">
                            <x-forms.label fieldId="eta" :fieldLabel="__('ETA')"></x-forms.label>
                            <input type="text" class="form-control height-35 f-15" name="eta" id="eta"
                                value="{{ $task->eta }}" placeholder="@lang('Estimated Time of Arrival')">
                        </div>
                    </div>
                    <!-- ETA END -->
                </div>

                <!-- CANCEL SAVE SEND START -->
                <div class="px-lg-4 px-md-4 px-3 py-3 c-inv-btns">
                    <button type="button" class="btn-cancel rounded mr-0 mr-lg-3 mr-md-3 f-15">
                        @lang('app.cancel')
                    </button>

                    <div class="d-flex">
                        <x-forms.button-primary class="save-form" icon="check">
                            @lang('app.save')
                        </x-forms.button-primary>
                    </div>
                </div>
                <!-- CANCEL SAVE SEND END -->
            </x-form>
            <!-- FORM END -->
        </div>
        <!-- EDIT TASK END -->
    </div>
@endsection

@push('scripts')
    <script>
        $(document).ready(function() {
            // Initialize date picker
            const dp1 = datepicker('#date', {
                position: 'bl',
                @if($task->date)
                    dateSelected: new Date("{{ str_replace('-', '/', $task->date->format('Y-m-d')) }}"),
                @endif
                ...datepickerConfig
            });

            // Form submission
            $('.save-form').click(function() {
                $.easyAjax({
                    url: "{{ route('tasksassignmentmodule.update', $task->id) }}",
                    container: '#saveTaskForm',
                    type: "POST",
                    blockUI: true,
                    redirect: true,
                    data: $('#saveTaskForm').serialize()
                });
            });

            // Cancel button
            $('.btn-cancel').click(function() {
                window.location.href = "{{ route('tasksassignmentmodule.index') }}";
            });
        });
    </script>
@endpush