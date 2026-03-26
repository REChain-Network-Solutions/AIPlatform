<div class="modal-header">
    <h5 class="modal-title" id="modelHeading">@lang('app.update') @lang('complaint::modules.complaintChannels')</h5>
    <button type="button"  class="close" data-dismiss="modal" aria-label="Close"><span aria-hidden="true">×</span></button>
</div>
<div class="modal-body">
    <div class="portlet-body">
        <x-form id="editTicketChannel" method="PUT" class="ajax-form">
            <div class="form-body">
                <div class="row">
                    <div class="col-lg-12">
                        <x-forms.text fieldId="channel_name" :fieldLabel="__('complaint::modules.complaintChannels')"
                            fieldName="channel_name" fieldRequired="true" fieldPlaceholder="e.g. Facebook, Phone, etc." :fieldValue="$channel->channel_name">
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
    // save ticket channnel
    $('#save-ticket-channel').click(function () {
        $.easyAjax({
            url: "{{route('complaintChannels.update', $channel->id)}}",
            container: '#editTicketChannel',
            type: "POST",
            blockUI: true,
            data: $('#editTicketChannel').serialize(),
            success: function (response) {
                if(response.status == 'success'){
                    window.location.reload();
                }
            }
        })
    });
</script>
