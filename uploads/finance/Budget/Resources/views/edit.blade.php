@extends('layouts.app')

@section('content')
    <div class="content-wrapper">
        <!-- EDIT BUDGET ALLOCATION START -->
        <div class="bg-white rounded b-shadow-4 create-inv">
            <!-- HEADING START -->
            <div class="px-lg-4 px-md-4 px-3 py-3">
                <h4 class="mb-0 f-21 font-weight-normal">@lang('Edit Budget Allocation')</h4>
            </div>
            <!-- HEADING END -->
            <hr class="m-0 border-top-grey">
            <!-- FORM START -->
            <x-form class="c-inv-form" id="saveBudgetForm" method="PUT">
                <!-- BASIC INFO START -->
                <div class="row px-lg-4 px-md-4 px-3 py-3">
                    <!-- PROJECT NAME -->
                    <div class="col-md-6">
                        <div class="form-group mb-4">
                            <x-forms.label fieldId="project_name" :fieldLabel="__('Project Name')" fieldRequired="true"></x-forms.label>
                            <input type="text" class="form-control height-35 f-15" name="project_name" id="project_name"
                                value="{{ $allocation->project_name }}" required>
                        </div>
                    </div>

                    <!-- DEPARTMENT -->
                    <div class="col-md-6">
                        <div class="form-group mb-4">
                            <x-forms.label fieldId="department" :fieldLabel="__('Department')" fieldRequired="true"></x-forms.label>
                            <input type="text" class="form-control height-35 f-15" name="department" id="department"
                                value="{{ $allocation->department }}" required>
                        </div>
                    </div>
                </div>

                <div class="row px-lg-4 px-md-4 px-3 py-3">
                    <!-- BUDGET REQUESTED -->
                    <div class="col-md-6">
                        <div class="form-group mb-4">
                            <x-forms.label fieldId="budget_requested" :fieldLabel="__('Requested Budget')" fieldRequired="true"></x-forms.label>
                            <div class="input-group">
                                <div class="input-group-prepend">
                                    <span class="input-group-text">{{ company()->currency->currency_symbol }}</span>
                                </div>
                                <input type="number" step="0.01" min="0" class="form-control height-35 f-15" 
                                    name="budget_requested" id="budget_requested" value="{{ $allocation->budget_requested }}" required>
                            </div>
                        </div>
                    </div>

                    <!-- BUDGET APPROVED -->
                    <div class="col-md-6">
                        <div class="form-group mb-4">
                            <x-forms.label fieldId="budget_approved" :fieldLabel="__('Approved Budget')"></x-forms.label>
                            <div class="input-group">
                                <div class="input-group-prepend">
                                    <span class="input-group-text">{{ company()->currency->currency_symbol }}</span>
                                </div>
                                <input type="number" step="0.01" min="0" class="form-control height-35 f-15" 
                                    name="budget_approved" id="budget_approved" value="{{ $allocation->budget_approved }}">
                            </div>
                        </div>
                    </div>
                </div>

                <div class="row px-lg-4 px-md-4 px-3 py-3">
                    <!-- APPROVAL STATUS -->
                    <div class="col-md-6">
                        <div class="form-group c-inv-select mb-4">
                            <x-forms.label fieldId="approval_status" :fieldLabel="__('Approval Status')" fieldRequired="true"></x-forms.label>
                            <div class="select-others height-35 rounded">
                                <select class="form-control select-picker" name="approval_status" id="approval_status" required>
                                    <option value="pending" @selected($allocation->approval_status == 'pending')>@lang('Pending')</option>
                                    <option value="approved" @selected($allocation->approval_status == 'approved')>@lang('Approved')</option>
                                    <option value="rejected" @selected($allocation->approval_status == 'rejected')>@lang('Rejected')</option>
                                </select>
                            </div>
                        </div>
                    </div>

                    <!-- APPROVAL DATE -->
                    <div class="col-md-6">
                        <div class="form-group mb-4">
                            <x-forms.label fieldId="approval_date" :fieldLabel="__('Approval Date')"></x-forms.label>
                            <div class="input-group">
                                <input type="text" id="approval_date" name="approval_date"
                                    class="px-6 position-relative text-dark font-weight-normal form-control height-35 rounded p-0 text-left f-15"
                                    placeholder="@lang('placeholders.date')"
                                    value="{{ $allocation->approval_date ? $allocation->approval_date->format(company()->date_format) : '' }}">
                            </div>
                        </div>
                    </div>
                </div>

                <div class="row px-lg-4 px-md-4 px-3 py-3">
                    <!-- COMMENTS -->
                    <div class="col-md-12">
                        <div class="form-group mb-4">
                            <x-forms.label fieldId="comments" :fieldLabel="__('Comments')"></x-forms.label>
                            <textarea class="form-control f-14 pt-2" rows="3" name="comments" id="comments"
                                placeholder="@lang('Add any comments here')">{{ $allocation->comments }}</textarea>
                        </div>
                    </div>
                </div>
                <!-- BASIC INFO END -->

                <!-- CANCEL SAVE SEND START -->
                <div class="px-lg-4 px-md-4 px-3 py-3 c-inv-btns">
                    <button type="button" class="btn-cancel rounded mr-0 mr-lg-3 mr-md-3 f-15">
                        @lang('app.cancel')
                    </button>

                    <div class="d-flex">
                        <x-forms.button-primary class="save-form" icon="check">
                            @lang('app.update')
                        </x-forms.button-primary>
                    </div>
                </div>
                <!-- CANCEL SAVE SEND END -->
            </x-form>
            <!-- FORM END -->
        </div>
        <!-- EDIT BUDGET ALLOCATION END -->
    </div>
@endsection

@push('scripts')
    <script>
        $(document).ready(function() {
            // Initialize date picker
            const dp1 = datepicker('#approval_date', {
                position: 'bl',
                @if($allocation->approval_date)
                    dateSelected: new Date("{{ str_replace('-', '/', $allocation->approval_date->format('Y-m-d')) }}"),
                @endif
                ...datepickerConfig
            });

            // Form submission
            $('.save-form').click(function() {
                $.easyAjax({
                    url: "{{ route('budgetallocationaprovalmodule.update', $allocation->id) }}",
                    container: '#saveBudgetForm',
                    type: "POST",
                    blockUI: true,
                    redirect: true,
                    data: $('#saveBudgetForm').serialize()
                });
            });

            // Cancel button
            $('.btn-cancel').click(function() {
                window.location.href = "{{ route('budgetallocationaprovalmodule.index') }}";
            });
        });
    </script>
@endpush