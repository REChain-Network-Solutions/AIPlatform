<div class="modal-header">
    <h5 class="modal-title" id="modelHeading">
        @lang('modules.inspection_schedules.payOffline')
    </h5>
    <button type="button"  class="close" data-dismiss="modal" aria-label="Close"><span
            aria-hidden="true">×</span></button>
</div>
    <div class="modal-body">
        <div class="portlet-body">
            <x-form id="offline-payment" method="POST" class="ajax-form">
                <input type="hidden" name="scheduleID" value="{{$scheduleID}}">
                <div class="form-body">
                    <div class="row" id="addressDetail">
                        <div class="col-lg-12 col-md-12">
                            <x-forms.select class="select-picker" fieldId="offlineMethod" :fieldLabel="__('modules.inspection_schedules.paymentMethod')"
                                fieldName="offlineMethod" search="true">
                                @foreach($methods as $method)
                                    <option value="{{ $method->id }}">{{ mb_ucwords($method->name) }}</option>
                                @endforeach
                            </x-forms.select>
                        </div>
                        <div class="col-lg-12 col-md-12">
                            <x-forms.file allowedFileExtensions="txt pdf doc xls xlsx docx rtf png jpg jpeg svg" class="mr-0 mr-lg-2 mr-md-2" :fieldLabel="__('app.receipt')"
                            fieldName="bill" fieldId="bill" :popover="__('messages.fileFormat.multipleImageFile')" />
                        </div>
                    </div>
                </div>
            </x-form>
        </div>
    </div>
    <div class="modal-footer">
        <x-forms.button-cancel data-dismiss="modal" class="border-0 mr-3">@lang('app.close')</x-forms.button-cancel>
        <x-forms.button-primary id="save-offline-payment" icon="check">@lang('app.save')</x-forms.button-primary>
    </div>

<script>
    $(".select-picker").selectpicker();

    $("#bill").dropify({
        messages: dropifyMessages
    });

    $('#save-offline-payment').click(function() {
        $.easyAjax({
            url: "{{ route('inspection_schedules.store_offline_payment') }}",
            container: '#offline-payment',
            type: "POST",
            redirect: true,
            file: true,
            data: $('#offline-payment').serialize()
        })
    })
</script>
