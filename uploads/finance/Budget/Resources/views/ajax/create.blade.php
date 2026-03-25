<x-form id="save-budget-allocation-form">
    <div class="row p-20">
        <div class="col-lg-12">
            <div class="add-client bg-white rounded">
                <h4 class="mb-0 p-20 f-21 font-weight-normal border-bottom-grey">
                    @lang('app.menu.add_budget_allocation')
                </h4>
                <div class="row p-20">
                    <div class="col-md-6">
                        <x-forms.text fieldId="project_id" :fieldLabel="__('modules.budget_allocation.projectId')"
                            fieldName="project_id" fieldRequired="true"
                            :fieldPlaceholder="__('placeholders.projectId')">
                        </x-forms.text>
                    </div>
                    <div class="col-md-6">
                        <x-forms.text fieldId="project_name" :fieldLabel="__('modules.budget_allocation.projectName')"
                            fieldName="project_name" fieldRequired="true"
                            :fieldPlaceholder="__('placeholders.projectName')">
                        </x-forms.text>
                    </div>

                    <div class="col-md-6">
                        <x-forms.text fieldId="department" :fieldLabel="__('app.department')" fieldName="department"
                            fieldRequired="true" :fieldPlaceholder="__('placeholders.department')">
                        </x-forms.text>
                    </div>

                    <div class="col-md-6">
                        <x-forms.number fieldId="budget_requested"
                            :fieldLabel="__('modules.budget_allocation.budgetRequested')" fieldName="budget_requested"
                            fieldRequired="true" :fieldPlaceholder="__('placeholders.amount')">
                        </x-forms.number>
                    </div>

                    <div class="col-md-6">
                        <x-forms.select fieldId="approval_status" :fieldLabel="__('app.status')"
                            fieldName="approval_status" fieldRequired="true">
                            <option value="pending">@lang('app.pending')</option>
                            <option value="approved">@lang('app.approved')</option>
                            <option value="rejected">@lang('app.rejected')</option>
                        </x-forms.select>
                    </div>

                    <div class="col-md-6" id="approved-amount-container" style="display: none;">
                        <x-forms.number fieldId="budget_approved"
                            :fieldLabel="__('modules.budget_allocation.budgetApproved')" fieldName="budget_approved"
                            :fieldPlaceholder="__('placeholders.amount')">
                        </x-forms.number>
                    </div>

                    <div class="col-md-6" id="approved-amount-container" style="display: none;">
                        <x-forms.number fieldId="budget_approved"
                            :fieldLabel="__('modules.budget_allocation.budgetApproved')" fieldName="budget_approved"
                            :fieldPlaceholder="__('placeholders.amount')">
                        </x-forms.number>
                    </div>

                    <div class="col-md-6" id="approval-date-container" style="display: none;">
                        <x-forms.datepicker fieldId="approval_date"
                            :fieldLabel="__('modules.budget_allocation.approvalDate')" fieldName="approval_date"
                            :fieldPlaceholder="__('placeholders.date')"
                            :fieldValue="now()->format(company()->date_format)" />
                    </div>

                    <div class="col-md-12">
                        <x-forms.textarea fieldId="comments" :fieldLabel="__('modules.budget_allocation.comments')"
                            fieldName="comments" :fieldPlaceholder="__('placeholders.comments')">
                        </x-forms.textarea>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <x-form-actions>
        <x-forms.button-primary id="save-budget-allocation" class="mr-3" icon="check">
            @lang('app.save')
        </x-forms.button-primary>
        <x-forms.button-cancel :link="route('budgetallocationaprovalmodule.index')" class="border-0">
            @lang('app.cancel')
        </x-forms.button-cancel>
    </x-form-actions>
</x-form>

<script>
    $(document).ready(function () {
        $('#approval_status').on('change', function() {
            if ($(this).val() === 'approved') {
                $('#approved-amount-container').show();
                $('#approval-date-container').show();
                $('#budget_approved').val($('#budget_requested').val());
                // Set default date if empty
                if (!$('#approval_date').val()) {
                    $('#approval_date').val("{{ now()->format(company()->date_format) }}");
                }
            } else {
                $('#approved-amount-container').hide();
                $('#approval-date-container').hide();
                $('#budget_approved').val('');
                $('#approval_date').val('');
            }
        });
        // Show/hide approved amount field based on status
        $('#approval_status').on('change', function () {
            if ($(this).val() === 'approved') {
                $('#approved-amount-container').show();
                $('#budget_approved').val($('#budget_requested').val());
            } else {
                $('#approved-amount-container').hide();
                $('#budget_approved').val('');
            }
        });

        // Set approved amount equal to requested when requested changes
        $('#budget_requested').on('change', function () {
            if ($('#approval_status').val() === 'approved') {
                $('#budget_approved').val($(this).val());
            }
        });

        $('#save-budget-allocation').click(function () {
            const url = "{{ route('budgetallocationaprovalmodule.store') }}";
            var data = $('#save-budget-allocation-form').serialize();

            $.easyAjax({
                url: url,
                container: '#save-budget-allocation-form',
                type: "POST",
                disableButton: true,
                blockUI: true,
                buttonSelector: "#save-budget-allocation",
                data: data,
                success: function (response) {
                    if (response.status == 'success') {
                        if (typeof MODAL_XL !== 'undefined' && $(MODAL_XL).hasClass('show')) {
                            $(MODAL_XL).modal('hide');
                            window.location.reload();
                        } else if (typeof response.redirectUrl !== 'undefined') {
                            setTimeout(function () {
                                $(MODAL_XL).modal('hide');
                            }, 1500);
                        }
                    }
                }
            });
        });
    });
</script>