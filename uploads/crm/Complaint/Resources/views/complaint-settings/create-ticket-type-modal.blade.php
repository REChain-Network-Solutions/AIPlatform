<div class="modal-header">
    <h5 class="modal-title">@lang('app.addNew') @lang('complaint::modules.complaintTypes')</h5>
    <button type="button" class="close" data-dismiss="modal" aria-hidden="true">×</button>
</div>

<div class="modal-body">
    <div class="portlet-body">
        <x-form id="createTicket" method="POST" class="ajax-form">
            <div class="form-group">
                <div class="row">
                    <div class="col-lg-12 ">
                        <x-forms.text fieldId="type" :fieldLabel="__('complaint::modules.complaintTypes')" fieldName="type"
                            fieldRequired="true" :fieldPlaceholder="__('placeholders.ticketType')">
                        </x-forms.text>
                    </div>
                </div>
            </div>
        </x-form>
    </div>
</div>
<div class="modal-footer">
    <x-forms.button-cancel data-dismiss="modal" class="border-0 mr-3">@lang('app.cancel')</x-forms.button-cancel>
    <x-forms.button-primary id="save-ticket-type" icon="check">@lang('app.save')</x-forms.button-primary>
</div>

<script>
    // save type
    $('#save-ticket-type').click(function() {
        $.easyAjax({
            url: "{{ route('complaintTypes.store') }}",
            container: '#createTicket',
            type: "POST",
            blockUI: true,
            data: $('#createTicket').serialize(),
            success: function(response) {
                if (response.status == "success") {
                    if ($('#ticket_type_id').length > 0) {
                        $('#ticket_type_id').html(response.optionData);
                        $('#ticket_type_id').selectpicker('refresh');
                        $(MODAL_LG).modal('hide');
                    } else {
                        window.location.reload();
                    }
                }
            }
        })
    });

</script>
