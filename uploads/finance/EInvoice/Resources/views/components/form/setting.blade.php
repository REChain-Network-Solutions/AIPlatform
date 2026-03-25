<div class="col-12">
    @php
        $companyAddressLink = '<a href="'.route('business-address.index').'">'.__('app.menu.businessAddresses').'</a>';
    @endphp
    <x-alert type="primary">
        <span class="mb-12"><strong>Note:</strong></span>
        <span>@lang('einvoice::app.companyAddressNote', ['link' => $companyAddressLink])</span>
    </x-alert>
</div>
<div class="col-lg-6">
    <x-forms.text class="mr-0 mr-lg-2 mr-md-2" :fieldLabel="__('einvoice::app.settingForm.electronicAddress')" fieldName="electronic_address" fieldId="electronic_address" :fieldValue="$setting?->electronic_address"/>
</div>
<div class="col-lg-6">
    <x-forms.text class="mr-0 mr-lg-2 mr-md-2" :fieldLabel="__('einvoice::app.settingForm.electronicAddressScheme')" fieldName="electronic_address_scheme" fieldId="electronic_address_scheme" :fieldValue="$setting?->electronic_address_scheme"/>
</div>
<div class="col-lg-6">
    <x-forms.text class="mr-0 mr-lg-2 mr-md-2" :fieldLabel="__('einvoice::app.settingForm.companyID')" fieldName="e_invoice_company_id" fieldId="e_invoice_company_id" :fieldValue="$setting?->e_invoice_company_id"/>
</div>
<div class="col-lg-6">
    <x-forms.text class="mr-0 mr-lg-2 mr-md-2" :fieldLabel="__('einvoice::app.settingForm.companyIDScheme')" fieldName="e_invoice_company_id_scheme" fieldId="e_invoice_company_id_scheme" :fieldValue="$setting?->e_invoice_company_id_scheme"/>
</div>
