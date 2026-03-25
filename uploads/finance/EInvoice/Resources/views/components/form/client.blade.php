<div class="col-12 p-0">
<h4 class="mb-0 py-20 pt-3 f-21 font-weight-normal  border-top-grey">
    @lang('einvoice::app.menu.einvoiceSettings')
</h4>
<div class="row py-20">
    <div class="col-lg-6">
        <x-forms.text class="mr-0 mr-lg-2 mr-md-2" :fieldLabel="__('einvoice::app.settingForm.electronicAddress')" fieldName="electronic_address"
            fieldId="electronic_address" :fieldValue="$clientDetails?->electronic_address" />
    </div>
    <div class="col-lg-6">
        <x-forms.text class="mr-0 mr-lg-2 mr-md-2" :fieldLabel="__('einvoice::app.settingForm.electronicAddressScheme')" fieldName="electronic_address_scheme"
            fieldId="electronic_address_scheme" :fieldValue="$clientDetails?->electronic_address_scheme" />
    </div>
</div>
</div>
