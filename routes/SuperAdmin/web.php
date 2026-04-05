<?php

use Illuminate\Support\Facades\Route;
use App\Http\Controllers\CustomModuleController;
use App\Http\Controllers\NotificationController;
use App\Http\Controllers\SuperAdmin\FaqController;
use App\Http\Controllers\SuperAdmin\MollieController;
use App\Http\Controllers\SuperAdmin\BillingController;
use App\Http\Controllers\SuperAdmin\CompanyController;
use App\Http\Controllers\SuperAdmin\InvoiceController;
use App\Http\Controllers\SuperAdmin\PackageController;
use App\Http\Controllers\SuperAdmin\PayFastController;
use App\Http\Controllers\SuperAdmin\PaystackController;
use App\Http\Controllers\SuperAdmin\AuthorizeController;
use App\Http\Controllers\SuperAdmin\DashboardController;
use App\Http\Controllers\SuperAdmin\PaypalIPNController;
use App\Http\Controllers\SuperAdmin\SuperAdminController;
use App\Http\Controllers\SuperAdmin\CustomFieldController;
use App\Http\Controllers\SuperAdmin\FaqCategoryController;
use App\Http\Controllers\SuperAdmin\ThemeSettingController;
use App\Http\Controllers\SuperAdmin\StripeWebhookController;
use App\Http\Controllers\SuperAdmin\InvoiceSettingController;
use App\Http\Controllers\SuperAdmin\PayFastWebhookController;
use App\Http\Controllers\SuperAdmin\ProfileSettingController;
use App\Http\Controllers\SuperAdmin\SupportTicketsController;
use App\Http\Controllers\SuperAdmin\PaystackWebhookController;
use App\Http\Controllers\SuperAdmin\RazorpayWebhookController;
use App\Http\Controllers\SuperAdmin\AuthorizeWebhookController;
use App\Http\Controllers\SuperAdmin\SuperAdminPaypalController;
use App\Http\Controllers\SuperAdmin\OfflinePlanChangeController;
use App\Http\Controllers\SuperAdmin\SupportTicketFileController;
use App\Http\Controllers\SuperAdmin\SupportTicketTypeController;
use App\Http\Controllers\SuperAdmin\SupportTicketReplyController;
use App\Http\Controllers\SuperAdmin\FrontSetting\SignUpController;
use App\Http\Controllers\SuperAdmin\GlobalCurrencySettingController;
use App\Http\Controllers\SuperAdmin\OfflinePaymentSettingController;
use App\Http\Controllers\SuperAdmin\FrontSetting\FrontMenuController;
use App\Http\Controllers\SuperAdmin\FrontSetting\SeoDetailController;
use App\Http\Controllers\SuperAdmin\FrontSetting\FaqSettingController;
use App\Http\Controllers\SuperAdmin\FrontSetting\FrontWidgetController;
use App\Http\Controllers\SuperAdmin\PaymentGatewayCredentialController;
use App\Http\Controllers\SuperAdmin\SuperadminRolePermissionController;
use App\Http\Controllers\SuperAdmin\FrontSetting\FrontSettingController;
use App\Http\Controllers\SuperAdmin\FrontSetting\ClientSettingController;
use App\Http\Controllers\SuperAdmin\FrontSetting\FooterSettingController;
use App\Http\Controllers\SuperAdmin\FrontSetting\FeatureSettingController;
use App\Http\Controllers\SuperAdmin\FrontSetting\SocialLinkSettingController;
use App\Http\Controllers\SuperAdmin\FrontSetting\TestimonialSettingController;
use App\Http\Controllers\SuperAdmin\FrontSetting\FeatureTranslationSettingController;
use App\Http\Controllers\SuperAdmin\FrontSetting\ThemeSettingController as FrontThemeSettingController;

Route::group(['middleware' => ['auth'], 'prefix' => 'account', 'as' => 'superadmin.'], function () {
    Route::get('impersonate/stop_impersonate', [SuperAdminController::class, 'stopImpersonate'])->name('superadmin.stop_impersonate');
    Route::get('workspaces', [SuperAdminController::class, 'workspaces'])->name('superadmin.workspaces');
    Route::post('choose-workspace', [SuperAdminController::class, 'chooseWorkspace'])->name('superadmin.choose_workspace');
    Route::get('refresh-cache', [\App\Http\Controllers\AppSettingController::class, 'refreshCache'])->name('superadmin.refresh-cache');
    Route::get('clearCache', [\App\Http\Controllers\AppSettingController::class, 'resetCache'])->name('superadmin.clear-cache');

    Route::post('signup/verifyEmail', [SignUpController::class, 'verifyEmail'])->name('signup.verifyEmail');
    Route::get('notify-admin', [NotificationController::class, 'notifyAdmin'])->name('notify.admin');
    Route::post('notify-admin', [NotificationController::class, 'notifyAdminSubmit'])->name('notify.admin.submit');
});

Route::group(['middleware' => ['auth', 'super-admin'], 'prefix' => 'account', 'as' => 'superadmin.'], function () {
    Route::get('checklists', [DashboardController::class, 'checklist'])->name('checklist');
    Route::get('super-admin-dashboard', [DashboardController::class, 'index'])->name('super_admin_dashboard');
    Route::get('companies/edit-package/{companyId}', [CompanyController::class, 'editPackage'])->name('companies.edit_package');
    Route::put('companies/edit-package/{companyId}', [CompanyController::class, 'updatePackage'])->name('companies.update_package');
    Route::post('companies/login_as_company/{companyId}', [CompanyController::class, 'loginAsCompany'])->name('companies.login_as_company');
    Route::post('companies/approve_company', [CompanyController::class, 'approveCompany'])->name('companies.approve_company');
    Route::resource('faqCategory', FaqCategoryController::class)->except(['index', 'edit', 'show']);
    Route::post('faqs/file-store', [FaqController::class, 'fileStore'])->name('faqs.file-store');
    Route::post('faqs/file-destroy/{id}', [FaqController::class, 'fileDelete'])->name('faqs.file-destroy');

    Route::resource('companies', CompanyController::class);
    Route::resource('superadmin-invoices', InvoiceController::class)->only(['index']);
    Route::resource('packages', PackageController::class)->except(['show']);

    Route::post('superadmin/assignRole', [SuperAdminController::class, 'assignRole'])->name('superadmin.assign_role');
    Route::resource('superadmin', SuperAdminController::class);

    Route::get('offline-plan/change-plan/{id}/{status}', [OfflinePlanChangeController::class, 'confirmChangePlan'])->name('offline-plan.confirmChangePlan');
    Route::post('offline-plan/change-plan', [OfflinePlanChangeController::class, 'changePlan'])->name('offline-plan.changePlan');
    Route::resource('offline-plan', OfflinePlanChangeController::class)->only(['index', 'show']);

    Route::post('support-tickets/apply-quick-action', [SupportTicketsController::class, 'applyQuickAction'])->name('support-tickets.apply_quick_action');
    Route::post('support-tickets/updateOtherData/{id}', [SupportTicketsController::class, 'updateOtherData'])->name('support-tickets.update_other_data');
    Route::get('company-ajax', [CompanyController::class, 'ajaxLoadCompany'])->name('get.company-ajax');

    Route::resource('support-ticketTypes', SupportTicketTypeController::class);

    Route::group(['prefix' => 'front-settings', 'as' => 'front-settings.'], function () {
        Route::get('/', function () {
            return redirect()->route('superadmin.front-settings.front_theme_settings');
        });
        Route::get('front-theme-settings', [FrontThemeSettingController::class, 'index'])->name('front_theme_settings');
        Route::put('front-theme-update', [FrontThemeSettingController::class, 'themeUpdate'])->name('front_theme_update');
        Route::get('front-settings-translation/{lang?}', [FrontSettingController::class, 'lang'])->name('front-settings.lang');
        Route::put('front-settings-translation', [FrontSettingController::class, 'updateLang'])->name('front-settings.update_lang');
        Route::get('price-settings-translation/{lang?}', [FrontSettingController::class, 'priceLang'])->name('price-settings.lang');
        Route::put('price-settings-translation', [FrontSettingController::class, 'updatePriceLang'])->name('price-settings.update_lang');

        Route::get('features-translation/{lang?}', [FeatureTranslationSettingController::class, 'lang'])->name('features-translation.lang');
        Route::put('features-translation', [FeatureTranslationSettingController::class, 'updateLang'])->name('features-translation.update_lang');

        Route::get('social-link', [SocialLinkSettingController::class, 'socialLink'])->name('social_link');
        Route::put('social-link', [SocialLinkSettingController::class, 'socialLinkUpdate'])->name('post.social_links');

        Route::resource('features-settings', FeatureSettingController::class);

        Route::put('footer-settings-translation', [FooterSettingController::class, 'updateLang'])->name('footer-settings.update_lang');
        Route::resource('footer-settings', FooterSettingController::class)->except(['index', 'show']);
        Route::post('footer-settings-slug', [FooterSettingController::class, 'generateSlug'])->name('footer-settings.generate_slug');
        Route::get('footer-settings/{lang?}', [FooterSettingController::class, 'index'])->name('footer-settings.index');

        Route::get('cta-settings/{lang?}', [FooterSettingController::class, 'ctaLang'])->name('cta-settings.lang');
        Route::put('cta-settings', [FooterSettingController::class, 'updateCtaLang'])->name('cta-settings.update_lang');
        Route::resource('front-widgets', FrontWidgetController::class);
        Route::get('seo-detail/{lang?}', [SeoDetailController::class, 'index'])->name('seo-detail.index');
        Route::resource('seo-detail', SeoDetailController::class)->only(['edit', 'update']);

        Route::get('contact-settings', [FrontSettingController::class, 'contact'])->name('contact_settings');
        Route::put('contact-settings', [FrontSettingController::class, 'contactUpdate'])->name('update_contact_settings');
        Route::get('auth-settings', [FrontSettingController::class, 'authSetting'])->name('auth_settings');
        Route::put('auth-settings', [FrontSettingController::class, 'authUpdate'])->name('auth_settings.update');
        Route::resource('front-settings', FrontSettingController::class)->only(['update']);
        Route::get('front-settings/{lang?}', [FrontSettingController::class, 'index'])->name('front-settings.index');

        Route::get('signup-setting-translation/{lang?}', [SignUpController::class, 'lang'])->name('signup_setting.lang');
        Route::put('signup-setting-translation', [SignUpController::class, 'updateLang'])->name('signup_setting.update_lang');
        Route::resource('sign-up-setting', SignUpController::class)->only(['index', 'update']);
        Route::get('testimonial-setting-translation', [TestimonialSettingController::class, 'createTestimonialTitle'])->name('create_testimonial_title');
        Route::post('testimonial-setting-translation', [TestimonialSettingController::class, 'storeOrUpdateTestimonialTitle'])->name('store_testimonial_title');
        Route::get('testimonial-setting-translation/{id}', [TestimonialSettingController::class, 'editTestimonialTitle'])->name('edit_testimonial_title');
        Route::resource('testimonial-settings', TestimonialSettingController::class);

        Route::put('client-setting-translation', [ClientSettingController::class, 'updateLang'])->name('client_setting.update_lang');
        Route::put('client-setting-translation', [ClientSettingController::class, 'updateLang'])->name('client_setting.update_lang');

        // Overwrite index and below added except
        Route::resource('client-settings', ClientSettingController::class)->except(['index', 'show']);
        Route::get('client-settings/{lang?}', [ClientSettingController::class, 'index'])->name('client-settings.index');

        Route::put('faq-setting-translation', [FaqSettingController::class, 'updateLang'])->name('faq_setting.update_lang');
        Route::resource('faq-settings', FaqSettingController::class)->except(['index', 'show']);
        Route::get('faq-settings/{lang?}', [FaqSettingController::class, 'index'])->name('faq-settings.index');

        Route::put('front-menu-settings', [FrontMenuController::class, 'updateLang'])->name('front_menu_settings.updateLang');
        Route::get('front-menu-settings/{lang?}', [FrontMenuController::class, 'lang'])->name('front_menu_settings.lang');
    });

    Route::group(['middleware' => 'auth', 'prefix' => 'settings', 'as' => 'settings.'], function () {
        Route::resource('super-admin-profile', ProfileSettingController::class);
        Route::resource('super-admin-theme-settings', ThemeSettingController::class);
        Route::resource('custom-module-settings', CustomModuleController::class);
        Route::resource('global-custom-fields', CustomFieldController::class);
        // Currency Settings routes
        Route::get('global-currency-settings/update/exchange-rates', [GlobalCurrencySettingController::class, 'updateExchangeRate'])->name('currency_settings.update_exchange_rates');

        /* Start Currency Settings routes */
        Route::get('global-currency-settings/exchange-key', [GlobalCurrencySettingController::class, 'currencyExchangeKey'])->name('currency_settings.exchange_key');
        Route::post('global-currency-settings/exchange-key-store', [GlobalCurrencySettingController::class, 'currencyExchangeKeyStore'])->name('currency_settings.exchange_key_store');
        Route::get('global-currency-settings/exchange-rate/{currency}', [GlobalCurrencySettingController::class, 'exchangeRate'])->name('currency_settings.exchange_rate');

        Route::get('global-currency-settings/update-currency-format', [GlobalCurrencySettingController::class, 'updateCurrencyFormat'])->name('currency_settings.update_currency_format');
        Route::resource('global-currency-settings', GlobalCurrencySettingController::class);
        Route::resource('global-payment-gateway-settings', PaymentGatewayCredentialController::class);
        Route::resource('global-offline-payment-setting', OfflinePaymentSettingController::class);
        Route::resource('global-invoice-settings', InvoiceSettingController::class);

        // SuperAdmin Role Permissions
        Route::post('superadmin-permissions/storeRole', [SuperadminRolePermissionController::class, 'storeRole'])->name('superadmin-permissions.store_role');
        Route::post('superadmin-permissions/deleteRole', [SuperadminRolePermissionController::class, 'deleteRole'])->name('superadmin-permissions.delete_role');
        Route::post('superadmin-permissions/permissions', [SuperadminRolePermissionController::class, 'permissions'])->name('superadmin-permissions.permissions');
        Route::post('superadmin-permissions/customPermissions', [SuperadminRolePermissionController::class, 'customPermissions'])->name('superadmin-permissions.custom_permissions');
        Route::post('superadmin-permissions/reset-permissions', [SuperadminRolePermissionController::class, 'resetPermissions'])->name('superadmin-permissions.reset_permissions');
        Route::resource('superadmin-permissions', SuperadminRolePermissionController::class);
    });
});

Route::group(['middleware' => ['auth', 'multi-company-select'], 'prefix' => 'account/settings'], function () {
    Route::get('billing/upgrade-plan', [BillingController::class, 'upgradePlan'])->name('billing.upgrade_plan');

    Route::post('billing/unsubscribe', [BillingController::class, 'cancelSubscription'])->name('billing.unsubscribe');
    Route::post('billing/razorpay-payment', [BillingController::class, 'razorpayPayment'])->name('billing.razorpay-payment');
    Route::post('billing/razorpay-subscription', [BillingController::class, 'razorpaySubscription'])->name('billing.razorpay-subscription');
    Route::get('billing/select-package/{packageID}', [BillingController::class, 'selectPackage'])->name('billing.select-package');
    Route::get('billing/package', [BillingController::class, 'packages'])->name('billing.packages');


    // Paypal IPN
    #Route::post('paypal-webhook/{hash}', [BillingController::class, 'webhook'])->name('paypal.webhook');
    #Route::get('paypal-webhook/{hash}', [BillingController::class, 'getWebhook'])->name('get_paypal.webhook');

    Route::post('billing/stripe-validate', [BillingController::class, 'stripeValidate'])->name('billing.stripe-validate');
    Route::post('billing/payment-stripe', [BillingController::class, 'payment'])->name('billing.stripe');
    Route::post('billing/payment-authorize', [AuthorizeController::class, 'createSubscription'])->name('billing.authorize');
    Route::post('billing/check-authorize-subscription', [AuthorizeController::class, 'checkSubscription'])->name('billing.check-authorize-subscription');
    Route::get('billing/invoice-download/{invoice}', [BillingController::class, 'download'])->name('billing.invoice-download');

    Route::get('billing/payfast-success', [PayFastController::class, 'payFastPaymentSuccess'])->name('billing.payfast-success');
    Route::get('billing/payfast-cancel', [PayFastController::class, 'payFastPaymentCancel'])->name('billing.payfast-cancel');

    Route::post('billing/payfast-pay', [PaystackController::class, 'redirectToGateway'])->name('billing.paystack');
    #Route::get('payfast/callback/{id}/{type}/{status}', [PayfastController::class, 'handleGatewayCallback'])->name('payfast.callback');
   # Route::post('payfast-webhook/{hash}', [PayfastController::class, 'handleGatewayWebhook'])->name('payfast.webhook');

    Route::post('billing/mollie', [MollieController::class, 'redirectToGateway'])->name('billing.mollie');

    Route::post('billing/free-plan', [BillingController::class, 'freePlan'])->name('billing.free-plan');

    Route::get('billing/paypal/{packageId}/{type}', [SuperAdminPaypalController::class, 'paymentWithpaypal'])->name('billing.paypal-payment');
    Route::get('billing/paypal-recurring', [SuperAdminPaypalController::class, 'payWithPaypalRecurrring'])->name('billing.paypal-recurring');
    Route::get('billing/paypal-invoice', [SuperAdminPaypalController::class, 'createInvoice'])->name('billing.paypal-invoice');
    Route::get('billing/paywithpaypal', [SuperAdminPaypalController::class, 'payWithPaypal'])->name('billing.paywithpaypal');
    Route::get('billing/cancel-agreement', [SuperAdminPaypalController::class, 'cancelAgreement'])->name('billing.cancel-agreement');


    Route::post('billing/lifetime', [BillingController::class, 'saveLifetimePackage'])->name('billing.lifetime');
    Route::post('billing/stripeNew/{companyId}', [BillingController::class, 'lifetimepaymentWithStripe'])->name('billing.stripeNew');

    // Route::get('paypal-public/{companyId}', [BillingController::class, 'paymentWithpaypalPublic'])->name('paypal_public');
    // Route::get('paypal/{companyId}', [BillingController::class, 'paymentWithpaypal'])->name('paypal');
    // Route::get('paypal', [BillingController::class, 'getPaymentStatus'])->name('get_paypal_status');
    // Route::get('paypal-recurring', [BillingController::class, 'payWithPaypalRecurring'])->name('paypal_recurring');



    // route for check status responce
    Route::get('billing/offline-payment', [BillingController::class, 'offlinePayment'])->name('billing.offline-payment');
    Route::post('billing/offline-payment-submit', [BillingController::class, 'offlinePaymentSubmit'])->name('billing.offline-payment-submit');


    Route::get('paypal', [SuperAdminPaypalController::class, 'getPaymentStatus'])->name('billing.paypal');


    Route::get('billing', [BillingController::class, 'index'])->name('billing.index');
});
Route::post('save-invoices', [StripeWebhookController::class, 'saveInvoices'])->name('billing.save_webhook');
Route::post('billing-verify-webhook/{id}', [StripeWebhookController::class, 'verifyStripeWebhook'])->name('billing.verify-webhook');
Route::post('save-razorpay-webhook/{id}', [RazorpayWebhookController::class, 'saveInvoices'])->name('billing.save_razorpay-webhook');
Route::post('save-paystack-webhook/{id}', [PaystackWebhookController::class, 'saveInvoices'])->name('billing.save_paystack-webhook');
Route::post('save-paypal-webhook/{id}', [PaypalIPNController::class, 'verpayment-authorizeifyBillingIPN'])->name('billing.save_paypal-webhook');
Route::post('save-authorize-webhook/{id}', [AuthorizeWebhookController::class, 'saveInvoices'])->name('billing.save_authorize-webhook');
Route::get('save-mollie-callback/{paymentId}/{hash}', [MollieController::class, 'handleGatewayCallback'])->name('billing.mollie.callback');
Route::post('save-mollie-webhook/{subscriptionId}/{hash}', [MollieController::class, 'handleGatewayWebhook'])->name('billing.mollie.webhook');
Route::post('payfast-notification/{id}', [PayFastWebhookController::class, 'saveInvoice'])->name('payfast-notification');
Route::get('billing/paystack/callback', [PaystackController::class, 'handleGatewayCallback'])->name('billing.paystack.callback');
Route::group(['middleware' => ['auth', 'admin-or-super-admin'], 'prefix' => 'account', 'as' => 'superadmin.'], function () {
    Route::get('superadmin-invoices/download/{id}', [InvoiceController::class, 'download'])->name('invoices.download');
    Route::get('offline-plan-files/download/{id}', [OfflinePlanChangeController::class, 'download'])->name('offline-plan.download');
    Route::get('billing-offline-plan-file/download/{id}', [BillingController::class, 'offlineFileDownload'])->name('billin-offline-plan.download');
    Route::get('faqs/searchquery/{query?}', [FaqController::class, 'searchQuery'])->name('faqs.searchQuery');
    Route::get('faqs/download/{id}', [FaqController::class, 'download'])->name('faqs.download');
    Route::resource('faqs', FaqController::class);
    Route::resource('support-tickets', SupportTicketsController::class);
    Route::get('support-ticket-files/download/{id}', [SupportTicketFileController::class, 'download'])->name('support-ticket-files.download');
    Route::resource('support-ticket-files', SupportTicketFileController::class);
    Route::resource('support-ticket-replies', SupportTicketReplyController::class);
});
