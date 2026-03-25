<x-form id="save-task-form">
    <div class="row p-20">
        <div class="col-lg-12">
            <div class="add-client bg-white rounded">
                <h4 class="mb-0 p-20 f-21 font-weight-normal border-bottom-grey">
                    @lang('Add New Task')
                </h4>
                <div class="row p-20">
                    <div class="col-md-6">
                        <x-forms.datepicker fieldId="date" 
                            :fieldLabel="__('Date')" 
                            fieldName="date"
                            fieldRequired="true"
                            :fieldPlaceholder="__('Select date')"
                            :fieldValue="now()->format('Y-m-d')"/>
                    </div>
                    
                    <div class="col-md-6">
                        <x-forms.text fieldId="task" 
                            :fieldLabel="__('Task')" 
                            fieldName="task"
                            fieldRequired="true" 
                            :fieldPlaceholder="__('Enter task description')">
                        </x-forms.text>
                    </div>
                    
                    <div class="col-md-6">
                        <x-forms.text fieldId="assign_to" 
                            :fieldLabel="__('Assign To')" 
                            fieldName="assign_to"
                            fieldRequired="true" 
                            :fieldPlaceholder="__('Person responsible')">
                        </x-forms.text>
                    </div>
                    
                    <div class="col-md-6">
                        <x-forms.select fieldId="product" 
                            :fieldLabel="__('Product')" 
                            fieldName="product"
                            fieldRequired="true">
                            <option value="">-- Select --</option>
                            <option value="WEBSITE">Website</option>
                            <option value="CRM">CRM</option>
                            <option value="BOTH">Both</option>
                        </x-forms.select>
                    </div>
                    
                    <div class="col-md-6">
                        <x-forms.select fieldId="priority" 
                            :fieldLabel="__('Priority')" 
                            fieldName="priority"
                            fieldRequired="true">
                            <option value="">-- Select --</option>
                            <option value="High">High</option>
                            <option value="Medium">Medium</option>
                            <option value="Low">Low</option>
                        </x-forms.select>
                    </div>
                    
                    <div class="col-md-6">
                        <x-forms.select fieldId="status" 
                            :fieldLabel="__('Status')" 
                            fieldName="status"
                            fieldRequired="true">
                            <option value="">-- Select --</option>
                            <option value="Not Yet Started">Not Yet Started</option>
                            <option value="In Progress">In Progress</option>
                            <option value="Completed">Completed</option>
                        </x-forms.select>
                    </div>
                    
                    <div class="col-md-12">
                        <x-forms.text fieldId="eta" 
                            :fieldLabel="__('ETA (Estimated Time of Arrival)')" 
                            fieldName="eta"
                            :fieldPlaceholder="__('e.g., 1 week, 24 hours')">
                        </x-forms.text>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <x-form-actions>
        <x-forms.button-primary id="save-task" class="mr-3" icon="check">
            @lang('Save')
        </x-forms.button-primary>
        <x-forms.button-cancel :link="route('tasksassignmentmodule.index')" class="border-0">
            @lang('Cancel')
        </x-forms.button-cancel>
    </x-form-actions>
</x-form>

<script>
$(document).ready(function() {
    // $('#date').datepicker({
    //     format: 'yyyy-mm-dd',
    //     autoclose: true,
    //     todayHighlight: true
    // });

    $('#save-task').click(function() {
        const url = "{{ route('tasksassignmentmodule.store') }}";
        const data = $('#save-task-form').serialize();

        $.easyAjax({
            url: url,
            container: '#save-task-form',
            type: "POST",
            disableButton: true,
            blockUI: true,
            buttonSelector: "#save-task",
            data: data,
            success: function(response) {
                if (response.status == 'success') {
                    if (typeof MODAL_XL !== 'undefined' && $(MODAL_XL).hasClass('show')) {
                        $(MODAL_XL).modal('hide');
                        window.location.reload();
                    } else if (response.redirectUrl) {
                        window.location.href = response.redirectUrl;
                    }
                }
            }
        });
    });
});
</script>