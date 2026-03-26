<div class="modal-header">
    <h5 class="modal-title" id="modelHeading">@lang('app.add') @lang('complaint::modules.complaintChannels')</h5>
    <button type="button"  class="close" data-dismiss="modal" aria-label="Close"><span
            aria-hidden="true">×</span></button>
</div>
<div class="modal-body">
    <div class="portlet-body">
        <x-form id="addTicketChannel" method="POST" class="ajax-form">
            <div class="form-body">
                <div class="row">
                    <div class="col-lg-12">
                        <x-forms.text fieldId="channel_name" :fieldLabel="__('complaint::modules.complaintChannels')"
                            fieldName="channel_name" fieldRequired="true" fieldPlaceholder="e.g. Facebook, Phone, etc.">
                        </x-forms.text>
                    </div>
                </div>
            </div>
        </x-form>
    </div>
</div>
<div class="modal-footer">
    <x-forms.button-cancel data-dismiss="modal" class="border-0 mr-3">@lang('app.close')</x-forms.button-cancel>
    <x-forms.button-primary id="save-ticket-channel" icon="check">@lang('app.save')</x-forms.button-primary>
</div>

<script>
    // save channel
    $('#save-ticket-channel').click(function() {
        $.easyAjax({
            url: "{{ route('complaintChannels.store') }}",
            container: '#addTicketChannel',
            type: "POST",
            blockUI: true,
            data: $('#addTicketChannel').serialize(),
            success: function(response) {
                if (response.status == "success") {
                    if ($('#ticket_channel_id').length > 0) {
                        $('#ticket_channel_id').html(response.optionData);
                        $('#ticket_channel_id').selectpicker('refresh');
                        $(MODAL_LG).modal('hide');
                    } else {
                        window.location.reload();
                    }
                }
            }
        })
    });

</script>
