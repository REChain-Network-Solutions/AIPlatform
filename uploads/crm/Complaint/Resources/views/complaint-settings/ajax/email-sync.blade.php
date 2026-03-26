<div class="col-lg-12 col-md-12 ntfcn-tab-content-left w-100 p-4 ">
    <div class="row">

        <div class="col-lg-12">
            <x-forms.checkbox :fieldLabel="__('app.status')" fieldName="status" fieldId="ticket_email_status"
                fieldValue="true" :checked="$ticketEmailSetting->status" />
        </div>

        <div class="col-lg-12 ticket_email_details @if (!$ticketEmailSetting->status) d-none @endif">
            <div class="row mt-3">
                <div class="col-md-12">
                    <h4 class="f-16 font-weight-500 text-capitalize">
                        @lang('modules.tickets.imapSettings')</h4>
                </div>

                <div class="col-md-4">
                    <x-forms.text class="mr-0 mr-lg-2 mr-md-2" :fieldLabel="__('modules.emailSettings.mailFrom')"
                        :fieldPlaceholder="__('placeholders.name')" fieldName="mail_from_name" fieldId="mail_from_name"
                        :fieldValue="$ticketEmailSetting->mail_from_name" :fieldRequired="true" />
                </div>

                <div class="col-md-4">
                    <x-forms.text class="mr-0 mr-lg-2 mr-md-2" :fieldLabel="__('modules.emailSettings.mailFromEmail')"
                        :fieldPlaceholder="__('placeholders.email')" fieldName="mail_from_email"
                        fieldId="mail_from_email" :fieldValue="$ticketEmailSetting->mail_from_email"
                        :fieldRequired="true" />
                </div>

                <div class="col-md-4">
                    <x-forms.text class="mr-0 mr-lg-2 mr-md-2" :fieldLabel="__('modules.emailSettings.mailUsername')"
                        :fieldPlaceholder="__('placeholders.email')" fieldName="mail_username" fieldId="mail_username"
                        :fieldValue="$ticketEmailSetting->mail_username" :fieldRequired="true" />
                </div>

                <div class="col-md-4">
                    <x-forms.password class="mr-0 mr-lg-2 mr-md-2"
                        :fieldLabel="__('modules.emailSettings.mailPassword')"
                        :fieldPlaceholder="__('modules.emailSettings.mailPassword')" fieldName="mail_password"
                        fieldId="mail_password" :fieldValue="$ticketEmailSetting->mail_password"
                        :fieldRequired="true" />
                </div>

                <div class="col-md-4">
                    <x-forms.text class="mr-0 mr-lg-2 mr-md-2" :fieldLabel="__('modules.tickets.imapHost')"
                        :fieldPlaceholder="__('modules.tickets.imapHost')" fieldName="imap_host" fieldId="imap_host"
                        :fieldValue="$ticketEmailSetting->imap_host" :fieldRequired="true" />
                </div>

                <div class="col-md-4">
                    <x-forms.text class="mr-0 mr-lg-2 mr-md-2" :fieldLabel="__('modules.tickets.imapPort')"
                        :fieldPlaceholder="__('modules.tickets.imapPort')" fieldName="imap_port" fieldId="imap_port"
                        :fieldValue="$ticketEmailSetting->imap_port" :fieldRequired="true" />
                </div>

                <div class="col-md-4">
                    <x-forms.text class="mr-0 mr-lg-2 mr-md-2" :fieldLabel="__('modules.tickets.imapEncryption')"
                        :fieldPlaceholder="__('modules.tickets.imapEncryption')" fieldName="imap_encryption"
                        fieldId="imap_encryption" :fieldValue="$ticketEmailSetting->imap_encryption" :fieldRequired="true" />
                </div>

                <div class="col-md-4">
                    <x-forms.label class="my-3" fieldId="sync_interval"
                        :fieldLabel="__('modules.tickets.syncIntervals')">
                    </x-forms.label>
                    <x-forms.input-group>
                        <input type="number" min="1" value="{{ $ticketEmailSetting->sync_interval }}" class="form-control height-35 f-14" name="sync_interval" id="sync_interval">

                        <x-slot name="append">
                            <span class="input-group-text height-35">@lang('app.minutes')</span>
                        </x-slot>
                    </x-forms.input-group>
                </div>

            </div>
        </div>

    </div>
</div>

<!-- Buttons Start -->
<div class="w-100 border-top-grey set-btns">
    <x-setting-form-actions>
        <x-forms.button-primary id="save-email-form" class="mr-3" icon="check">@lang('app.save')
        </x-forms.button-primary>
    </x-setting-form-actions>
</div>
<!-- Buttons End -->

<script>
    $(document).on('change', '#ticket_email_status', function() {
        $('.ticket_email_details').toggleClass('d-none');
    });

    $('body').on('click', '#save-email-form', function() {
        var url = "{{ route('complaint-email-settings.update', $ticketEmailSetting) }}";

        $.easyAjax({
            url: url,
            type: "POST",
            container: "#editSettings",
            blockUI: true,
            data: $('#editSettings').serialize(),
        })
    });

    $('body').on('click', '#send-test-notification', function() {
        var url = '{{ route('slack_settings.send_test_notification') }}';
        $.easyAjax({
            url: url,
            type: "GET",
        })
    });
</script>
