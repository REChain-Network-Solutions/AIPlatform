<div class="modal-header">
    <h5 class="modal-title" id="modelHeading">@lang('einvoice::app.menu.einvoiceSettings')</h5>
    <button type="button"  class="close" data-dismiss="modal" aria-label="Close"><span
            aria-hidden="true">×</span></button>
</div>
<div class="modal-body">

    <x-form id="editClinetSettings" method="PUT">
        <div class="row justify-content-between">
            <div class="col-lg-6">
                <x-forms.text class="mr-0 mr-lg-2 mr-md-2" :fieldLabel="__('einvoice::app.settingForm.electronicAddress')" fieldName="electronic_address"
                    fieldId="electronic_address" :fieldValue="$clientDetails?->electronic_address" />
            </div>
            <div class="col-lg-6">
                <x-forms.text class="mr-0 mr-lg-2 mr-md-2" :fieldLabel="__('einvoice::app.settingForm.electronicAddressScheme')" fieldName="electronic_address_scheme"
                    fieldId="electronic_address_scheme" :fieldValue="$clientDetails?->electronic_address_scheme" />
            </div>
        </div>
    </x-form>

</div>
<div class="modal-footer">
    <x-forms.button-cancel data-dismiss="modal" class="border-0 mr-3">@lang('app.close')</x-forms.button-cancel>
    <x-forms.button-primary id="save-client-form" icon="check">@lang('app.save')</x-forms.button-primary>
</div>

<script>
    $('body').on('click', '#save-client-form', function() {
        $.easyAjax({
            url: "{{ route('einvoice.client_save', $clientDetails?->id) }}",
            container: '#editClinetSettings',
            type: "POST",
            redirect: true,
            file: true,
            data: $('#editClinetSettings').serialize(),
            disableButton: true,
            blockUI: true,
            buttonSelector: "#save-client-form",
            success: function(response) {
                if (response.status == 'success') {
                    window.location.reload();
                }
            }
        })
    });
</script>

