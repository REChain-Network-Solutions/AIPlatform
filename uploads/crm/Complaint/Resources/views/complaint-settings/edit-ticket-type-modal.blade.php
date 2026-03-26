<div class="modal-header">
    <h5 class="modal-title" id="modelHeading">@lang('app.update') @lang('complaint::modules.complaintTypes')</h5>
    <button type="button"  class="close" data-dismiss="modal" aria-label="Close"><span
            aria-hidden="true">×</span>
    </button>
</div>

<div class="modal-body">
    <div class="portlet-body">
        <x-form id="editTicketType" method="PUT" class="ajax-form">
            <div class="form-body">
                <div class="row">
                    <div class="col-lg-12">
                        <x-forms.text fieldId="type" :fieldLabel="__('complaint::modules.complaintTypes')"
                            fieldName="type" fieldRequired="true" :fieldPlaceholder="__('placeholders.ticketType')" :fieldValue="$type->type">
                        </x-forms.text>
                    </div>
                </div>
            </div>
        </x-form>
    </div>
</div>

<div class="modal-footer">
    <x-forms.button-cancel data-dismiss="modal" class="border-0 mr-3">@lang('app.close')</x-forms.button-cancel>
    <x-forms.button-primary id="save-group" icon="check">@lang('app.save')</x-forms.button-primary>
</div>

<script>
    $('#editTicketType').on('submit', function(e) {
        return false;
    })

    $('#save-group').click(function () {
        $.easyAjax({
            url: "{{route('complaintTypes.update', $type->id)}}",
            container: '#editTicketType',
            type: "POST",
            data: $('#editTicketType').serialize(),
            success: function (response) {
                if(response.status == 'success'){
                    window.location.reload();
                }
            }
        })
    });
</script>
