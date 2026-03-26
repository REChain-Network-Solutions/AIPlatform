<div class="modal-header">
    <h5 class="modal-title">@lang('app.addNew') @lang('complaint::modules.complaintAgents')</h5>
    <button type="button" class="close" data-dismiss="modal" aria-hidden="true">×</button>
</div>
<div class="modal-body">
    <div class="portlet-body">
        <x-form id="createMethods" method="POST" class="ajax-form">
            <div class="row">
                <div class="col-md-6">
                    <x-forms.select fieldId="user_id" :fieldLabel="__('modules.tickets.chooseAgents')"
                        fieldName="user_id[]" search="true" fieldRequired="true" multiple="true">
                        @foreach ($employees as $emp)
                            <x-user-option :user="$emp" />
                        @endforeach
                    </x-forms.select>
                </div>

                <div class="col-md-6">
                    <x-forms.label class="mt-3" fieldId="category_id" fieldRequired="true"
                        :fieldLabel="__('modules.tickets.assignGroup')">
                    </x-forms.label>
                    <x-forms.input-group>
                        <select class="form-control select-picker" id="group_id" name="group_id"
                            data-live-search="true">
                            @foreach ($groups as $group)
                                <option value="{{ $group->id }}">{{ mb_ucwords($group->group_name) }}</option>
                            @endforeach
                        </select>
                        <x-slot name="append">
                            <button id="manage-groups" type="button"
                                class="btn btn-outline-secondary border-grey">@lang('app.add')</button>
                        </x-slot>
                    </x-forms.input-group>
                </div>
            </div>
        </x-form>
    </div>
</div>
<div class="modal-footer">
    <x-forms.button-cancel data-dismiss="modal" class="border-0 mr-3">@lang('app.cancel')</x-forms.button-cancel>
    <x-forms.button-primary id="save-agent" icon="check">@lang('app.save')</x-forms.button-primary>
</div>

<script>
    $(".select-picker").selectpicker();

    // save agent
    $('#save-agent').click(function() {
        $.easyAjax({
            url: "{{ route('complaint-agents.store') }}",
            container: '#createMethods',
            type: "POST",
            blockUI: true,
            data: $('#createMethods').serialize(),
            success: function(response) {
                if (response.status == "success") {
                    window.location.reload();
                }
            }
        })
    });

    $('#manage-groups').click(function() {
        var url = "{{ route('complaint-groups.create') }}";
        $(MODAL_LG + ' ' + MODAL_HEADING).html('...');
        $.ajaxModal(MODAL_LG, url);
    });
</script>
