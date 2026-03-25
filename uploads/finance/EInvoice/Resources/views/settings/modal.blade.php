<div class="modal-header">
    <h5 class="modal-title" id="modelHeading">@lang('einvoice::app.menu.einvoiceSettings')</h5>
    <button type="button"  class="close" data-dismiss="modal" aria-label="Close"><span
            aria-hidden="true">×</span></button>
</div>
<div class="modal-body">

    <x-form id="editSettings" method="PUT">
        <div class="row justify-content-between">
            <x-einvoice::form.setting />
        </div>
    </x-form>

</div>
<div class="modal-footer">
    <x-forms.button-cancel data-dismiss="modal" class="border-0 mr-3">@lang('app.close')</x-forms.button-cancel>
    <x-forms.button-primary id="save-form" icon="check">@lang('app.save')</x-forms.button-primary>
</div>

@include('einvoice::settings.save-script')
