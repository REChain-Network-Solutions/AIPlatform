<?php

/* Setting menu routes starts from here */
use App\Http\Controllers\SettingsController;
use App\Http\Controllers\ProfileController;
use App\Http\Controllers\LeaveTypeController;
use App\Http\Controllers\AppSettingController;
use App\Http\Controllers\GoogleAuthController;
use App\Http\Controllers\TicketTypeController;
use App\Http\Controllers\CustomFieldController;
use App\Http\Controllers\LeadSettingController;
use App\Http\Controllers\SmtpSettingController;
use App\Http\Controllers\TaskSettingController;
use App\Http\Controllers\TicketAgentController;
use App\Http\Controllers\TicketGroupController;
use App\Http\Controllers\CustomModuleController;
use App\Http\Controllers\LeaveSettingController;
use App\Http\Controllers\SlackSettingController;
use App\Http\Controllers\ThemeSettingController;
use App\Http\Controllers\TwoFASettingController;
use App\Http\Controllers\EmployeeShiftController;
use App\Http\Controllers\ModuleSettingController;
use App\Http\Controllers\TicketChannelController;
use App\Http\Controllers\TicketSettingController;
use App\Http\Controllers\InvoiceSettingController;
use App\Http\Controllers\MessageSettingController;
use App\Http\Controllers\ProfileSettingController;
use App\Http\Controllers\ProjectSettingController;
use App\Http\Controllers\PusherSettingsController;
use App\Http\Controllers\RolePermissionController;
use App\Http\Controllers\StorageSettingController;
use App\Http\Controllers\TimeLogSettingController;
use App\Http\Controllers\BusinessAddressController;
use App\Http\Controllers\CurrencySettingController;
use App\Http\Controllers\LanguageSettingController;
use App\Http\Controllers\SecuritySettingController;
use App\Http\Controllers\LeadAgentSettingController;
use App\Http\Controllers\PushNotificationController;
use App\Http\Controllers\AttendanceSettingController;
use App\Http\Controllers\ContractSettingController;
use App\Http\Controllers\CustomLinkSettingController;
use App\Http\Controllers\LeadSourceSettingController;
use App\Http\Controllers\SocialAuthSettingController;
use App\Http\Controllers\TicketEmailSettingController;
use App\Http\Controllers\TicketReplyTemplatesController;
use App\Http\Controllers\DatabaseBackupSettingController;
use App\Http\Controllers\GoogleCalendarSettingController;
use App\Http\Controllers\LeadPipelineSettingController;
use App\Http\Controllers\LeadStageSettingController;
use App\Http\Controllers\OfflinePaymentSettingController;
use App\Http\Controllers\PaymentGatewayCredentialController;
use App\Http\Controllers\NotificationSettingController;
use App\Http\Controllers\QuickbookSettingsController;
use App\Http\Controllers\ShiftRotationController;
use App\Http\Controllers\SignUpSettingController;
use App\Http\Controllers\TaxSettingController;
use App\Http\Controllers\UnitTypeController;
use App\Http\Controllers\UpdateAppController;
use Illuminate\Support\Facades\Route;

Route::get('account/settings/google-auth', [GoogleAuthController::class, 'index'])->name('googleAuth');

Route::group(['middleware' => 'auth', 'prefix' => 'account/settings'], function () {

    Route::post('app-settings/deleteSessions', [AppSettingController::class, 'deleteSessions'])->name('app-settings.delete_sessions');
    Route::resource('app-settings', AppSettingController::class);
    Route::resource('profile-settings', ProfileSettingController::class);

    /* 2FA */
    Route::get('2fa-codes-download', [TwoFASettingController::class, 'download'])->name('2fa_codes_download');
    Route::get('verify-2fa-password', [TwoFASettingController::class, 'verify'])->name('verify_2fa_password');
    Route::get('2fa-confirm', [TwoFASettingController::class, 'showConfirm'])->name('two-fa-settings.validate_confirm');
    Route::post('2fa-confirm', [TwoFASettingController::class, 'confirm'])->name('two-fa-settings.confirm');
    Route::get('2fa-email-confirm', [TwoFASettingController::class, 'showEmailConfirm'])->name('two-fa-settings.validate_email_confirm');
    Route::post('2fa-email-confirm', [TwoFASettingController::class, 'emailConfirm'])->name('two-fa-settings.email_confirm');
    Route::resource('two-fa-settings', TwoFASettingController::class);

    Route::post('profile/dark-theme', [ProfileController::class, 'darkTheme'])->name('profile.dark_theme');
    Route::post('profile/updateOneSignalId', [ProfileController::class, 'updateOneSignalId'])->name('profile.update_onesignal_id');
    Route::resource('profile', ProfileController::class);

    Route::get('smtp-settings/show-send-test-mail-modal', [SmtpSettingController::class, 'showTestEmailModal'])->name('smtp_settings.show_send_test_mail_modal');
    Route::get('smtp-settings/send-test-mail', [SmtpSettingController::class, 'sendTestEmail'])->name('smtp_settings.send_test_mail');

    Route::get('slack-settings/send-test-notification', [SlackSettingController::class, 'sendTestNotification'])->name('slack_settings.send_test_notification');

    Route::get('push-notification-settings/send-test-notification', [PushNotificationController::class, 'sendTestNotification'])->name('push_notification_settings.send_test_notification');

    Route::resource('smtp-settings', SmtpSettingController::class);
    Route::resource('notifications', NotificationSettingController::class);
    Route::resource('slack-settings', SlackSettingController::class);
    Route::resource('push-notification-settings', PushNotificationController::class);
    Route::resource('pusher-settings', PusherSettingsController::class);

    // Currency Settings routes
    Route::get('currency-settings/update/exchange-rates', [CurrencySettingController::class, 'updateExchangeRate'])->name('currency_settings.update_exchange_rates');

    /* Start Currency Settings routes */
    Route::get('currency-settings/exchange-key', [CurrencySettingController::class, 'currencyExchangeKey'])->name('currency_settings.exchange_key');
    Route::post('currency-settings/exchange-key-store', [CurrencySettingController::class, 'currencyExchangeKeyStore'])->name('currency_settings.exchange_key_store');
    Route::get('currency-settings/exchange-rate/{currency}', [CurrencySettingController::class, 'exchangeRate'])->name('currency_settings.exchange_rate');

    Route::get('currency-settings/update-currency-format', [CurrencySettingController::class, 'updateCurrencyFormat'])->name('currency_settings.update_currency_format');
    Route::resource('currency-settings', CurrencySettingController::class);
    Route::resource('payment-gateway-settings', PaymentGatewayCredentialController::class);
    /* End Currency Settings routes */

    Route::resource('offline-payment-setting', OfflinePaymentSettingController::class);

    /* Invoice Setting Routes */
    Route::post('invoice-settings/update-template/{id}', [InvoiceSettingController::class, 'updateTemplate'])->name('invoice_settings.update_template');
    Route::post('invoice-settings/update-prefix/{id}', [InvoiceSettingController::class, 'updatePrefix'])->name('invoice_settings.update_prefix');
    Route::resource('invoice-settings', InvoiceSettingController::class);

    /* unitType */
    Route::resource('unit-type', UnitTypeController::class);
    Route::post('unit-types/set-default', [UnitTypeController::class, 'setDefaultUnit'])->name('unit-type.set_default');

    /* Start Ticket settings routes */
    Route::post('ticket-agents/update-group/{id}', [TicketAgentController::class, 'updateGroup'])->name('ticket_agents.update_group');
    Route::resource('ticket-agents', TicketAgentController::class);
    Route::get('agent-groups', [TicketAgentController::class, 'agentGroups'])->name('ticket_agents.agent_groups');

    Route::resource('ticket-settings', TicketSettingController::class);
    Route::post('ticket-settings-status/update-status/{companyId}', [TicketSettingController::class, 'updateTicketSettingStatus'])->name('ticket-setting.update_status');
    Route::post('/ticket-agent-settings/{companyId}', [TicketSettingController::class, 'updateTicketSettingForAgent'])->name('ticket-agent-settings.update');
    Route::resource('ticket-groups', TicketGroupController::class);
    Route::resource('ticketTypes', TicketTypeController::class);
    Route::resource('ticketChannels', TicketChannelController::class);
    Route::resource('ticket-email-settings', TicketEmailSettingController::class);

    Route::get('replyTemplates/fetch-template', [TicketReplyTemplatesController::class, 'fetchTemplate'])->name('replyTemplates.fetchTemplate');
    Route::resource('replyTemplates', TicketReplyTemplatesController::class);
    /* End Ticket settings routes */
    Route::get('project-settings/create-category', [ProjectSettingController::class, 'createCategory'])->name('project-settings.createCategory');
    Route::post('project-settings/save-project-category', [ProjectSettingController::class, 'saveProjectCategory'])->name('project-settings.saveProjectCategory');
    Route::resource('project-settings', ProjectSettingController::class);
    Route::post('project-settings/{id?}', [ProjectSettingController::class, 'statusUpdate'])->name('project-settings.statusUpdate');
    Route::put('project-settings/change-status/{id?}', [ProjectSettingController::class, 'changeStatus'])->name('project-settings.changeStatus');
    Route::post('project-settings/set-default/{id?}', [ProjectSettingController::class, 'setDefault'])->name('project-settings.setDefault');
    // Route::get('check-qr-login', [AttendanceSettingController::class, 'qrClockInOut'])->name('settings.qr-login');
    // Route::post('change-qr-code-status', [AttendanceSettingController::class, 'qrCodeStatus'])->name('settings.change-qr-code-status');

    Route::put('attendance-settings/{id}/regularisation', [AttendanceSettingController::class, 'attendanceRegularisation'])->name('attendance-settings.attendanceRegularisation');
    Route::resource('attendance-settings', AttendanceSettingController::class);

    // Shift Rotation routes
    Route::post('shift-rotations/change-status', [ShiftRotationController::class, 'changeStatus'])->name('shift-rotations.change_status');
    Route::get('shift-rotations/automate-shift', [ShiftRotationController::class, 'automateShift'])->name('shift-rotations.automate_shift');
    Route::post('shift-rotations/remove_employee', [ShiftRotationController::class, 'removeEmployee'])->name('shift-rotations.remove_employee');
    Route::post('shift-rotations/store-automate-shift', [ShiftRotationController::class, 'storeAutomateShift'])->name('shift-rotations.store_automate_shift');
    Route::get('shift-rotations/manage-rotation-employee/{id}', [ShiftRotationController::class, 'manageEmployees'])->name('shift-rotations.manage_rotation_employee');
    Route::post('shift-rotations/change-employee-rotation', [ShiftRotationController::class, 'changeEmployeeRotation'])->name('shift-rotations.change_employee_rotation');
    Route::get('shift-rotations/run-rotation', [ShiftRotationController::class, 'runRotation'])->name('shift-rotations.run_rotation_get');
    Route::post('shift-rotations/run-rotation', [ShiftRotationController::class, 'runRotation'])->name('shift-rotations.run_rotation_post');
    Route::resource('shift-rotations', ShiftRotationController::class);

    Route::resource('leaves-settings', LeaveSettingController::class);
    Route::post('leaves-settings/change-permission', [LeaveSettingController::class, 'changePermission'])->name('leaves-settings.changePermission');

    // LeaveType Resource
    Route::resource('leaveType', LeaveTypeController::class);

    // Custom Fields Settings
    Route::resource('custom-fields', CustomFieldController::class);

    // Tax Settings
    Route::put('taxes/{tax}/to-keep-under-total-payment', [TaxSettingController::class, 'updateToKeepUnderTotalPayment'])->name('taxes.update_to_keep_under_total_payment');
    Route::resource('taxes', TaxSettingController::class);

    // Message settings
    Route::resource('message-settings', MessageSettingController::class);

    // Storage settings
    Route::get('storage-settings/aws-local-to-aws-modal', [StorageSettingController::class, 'awsLocalToAwsModal'])->name('storage-settings.aws_local_to_aws_modal');
    Route::post('storage-settings/aws-local-to-aws', [StorageSettingController::class, 'moveFilesLocalToAwsS3'])->name('storage-settings.aws_local_to_aws');
    Route::get('storage-settings/storage-test-modal/{type}', [StorageSettingController::class, 'awsTestModal'])->name('storage-settings.aws_test_modal');
    Route::post('storage-settings/aws-test', [StorageSettingController::class, 'awsTest'])->name('storage-settings.aws_test');
    Route::resource('storage-settings', StorageSettingController::class);

    // Language settings
    Route::get('language-settings/auto-translate', [LanguageSettingController::class, 'autoTranslate'])->name('language_settings.auto_translate');
    Route::post('language-settings/auto-translate', [LanguageSettingController::class, 'autoTranslateUpdate'])->name('language_settings.auto_translate_update');
    Route::post('language-settings/update-data/{id?}', [LanguageSettingController::class, 'updateData'])->name('language_settings.update_data');
    Route::post('language-settings/fix-translation', [LanguageSettingController::class, 'fixTranslation'])->name('language_settings.fix_translation');
    Route::post('language-settings/create-en-locale', [LanguageSettingController::class, 'createEnLocale'])->name('language_settings.create_en_locale');
    Route::resource('language-settings', LanguageSettingController::class);

    // Task Settings
    Route::resource('task-settings', TaskSettingController::class, ['only' => ['index', 'store']]);

    // Time Log Settings
    Route::resource('timelog-settings', TimeLogSettingController::class);

    // Social Auth Settings
    Route::resource('social-auth-settings', SocialAuthSettingController::class, ['only' => ['index', 'update']]);

    /* Lead Settings */
    Route::resource('lead-settings', LeadSettingController::class);
    Route::post('lead-settings-status/update-status/{companyId}', [LeadSettingController::class, 'updateLeadSettingStatus'])->name('lead-setting.update_status');
    Route::resource('lead-source-settings', LeadSourceSettingController::class);

    Route::get('lead-stage-update/{statusId}', [LeadStageSettingController::class, 'statusUpdate'])->name('lead-stage-setting.stageUpdate');
    Route::resource('lead-stage-setting', LeadStageSettingController::class);

    Route::get('lead-pipeline-update/{statusId}', [LeadPipelineSettingController::class, 'statusUpdate'])->name('lead-pipeline-update.stageUpdate');
    Route::resource('lead-pipeline-setting', LeadPipelineSettingController::class);

    Route::resource('lead-agent-settings', LeadAgentSettingController::class);
    Route::post('lead-agent-settings/update-category/{id}', [LeadAgentSettingController::class, 'updateCategory'])->name('lead_agents.update_category');
    Route::post('lead-agent-settings/update-status/{id}', [LeadAgentSettingController::class, 'updateStatus'])->name('lead_agents.update_status');
    Route::get('agent-category', [LeadAgentSettingController::class, 'agentCategories'])->name('lead_agent.categories');

    /* Contract Setting */
    Route::resource('contract-settings', ContractSettingController::class);

    // Security Settings
    Route::get('verify-google-recaptcha-v3', [SecuritySettingController::class, 'verify'])->name('verify_google_recaptcha_v3');
    Route::resource('security-settings', SecuritySettingController::class);

    // Google Calendar Settings
    Route::resource('google-calendar-settings', GoogleCalendarSettingController::class);
    Route::delete('google-auth', [GoogleAuthController::class, 'destroy'])->name('googleAuth.destroy');


    // Database Backup Settings
    Route::get('database-backup-settings/create-backup', [DatabaseBackupSettingController::class, 'createBackup'])->name('database-backup-settings.create_backup');
    Route::get('database-backup-settings/download/{file_name}', [DatabaseBackupSettingController::class, 'download'])->name('database-backup-settings.download');
    Route::get('database-backup-settings/delete/{file_name}', [DatabaseBackupSettingController::class, 'delete'])->name('database-backup-settings.delete');
    Route::resource('database-backup-settings', DatabaseBackupSettingController::class);

    // Role Permissions
    Route::post('role-permission/storeRole', [RolePermissionController::class, 'storeRole'])->name('role-permissions.store_role');
    Route::post('role-permission/deleteRole', [RolePermissionController::class, 'deleteRole'])->name('role-permissions.delete_role');
    Route::post('role-permissions/permissions', [RolePermissionController::class, 'permissions'])->name('role-permissions.permissions');
    Route::post('role-permissions/customPermissions', [RolePermissionController::class, 'customPermissions'])->name('role-permissions.custom_permissions');
    Route::post('role-permissions/reset-permissions', [RolePermissionController::class, 'resetPermissions'])->name('role-permissions.reset_permissions');
    Route::resource('role-permissions', RolePermissionController::class);

    // Theme settings
    Route::resource('theme-settings', ThemeSettingController::class);

    // Module settings
    Route::resource('module-settings', ModuleSettingController::class);

    // Custom Modules
    Route::post('custom-modules/verify-purchase', [CustomModuleController::class, 'verifyingModulePurchase'])->name('custom-modules.verify_purchase');
    Route::post('custom-modules/analyze', [CustomModuleController::class, 'analyze'])->name('custom-modules.analyze');
    Route::post('custom-modules/analyze-download', [CustomModuleController::class, 'downloadAnalysis'])->name('custom-modules.analyze_download');
    Route::post('custom-modules/apply-safe-fixes', [CustomModuleController::class, 'applySafeFixes'])->name('custom-modules.apply_safe_fixes');
    Route::resource('custom-modules', CustomModuleController::class);

    Route::post('business-address/set-default', [BusinessAddressController::class, 'setDefaultAddress'])->name('business-address.set_default');
    Route::resource('business-address', BusinessAddressController::class);

    Route::post('employee-shifts/set-default', [EmployeeShiftController::class, 'setDefaultShift'])->name('employee-shifts.set_default');
    Route::resource('employee-shifts', EmployeeShiftController::class);

    Route::resource('quickbooks-settings', QuickbookSettingsController::class);

    Route::resource('custom-link-settings', CustomLinkSettingController::class);

    Route::resource('sign-up-settings', SignUpSettingController::class)->only(['index', 'update']);

});

Route::group(['middleware' => 'auth', 'prefix' => 'account'], function () {

    Route::resource('company-settings', SettingsController::class)->only(['edit', 'update', 'index', 'change_language']);

    // Update App
    Route::post('update-settings/deleteFile', [UpdateAppController::class, 'deleteFile'])->name('update-settings.deleteFile');
    Route::get('update-settings/install', [UpdateAppController::class, 'install'])->name('update-settings.install');
    Route::resource('update-settings', UpdateAppController::class);

});

// Pass 9: user menu visibility diagnostics
