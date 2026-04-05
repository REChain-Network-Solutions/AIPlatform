<!-- SETTINGS SIDEBAR START -->
<div class="mobile-close-overlay w-100 h-100" id="close-settings-overlay"></div>
<div class="settings-sidebar bg-white py-3" id="mob-settings-sidebar">
    <a class="d-block d-lg-none close-it" id="close-settings"><i class="fa fa-times"></i></a>

    <!-- SETTINGS SEARCH START -->
    <form class="border-bottom-grey px-4 pb-3 d-flex">
        <div class="input-group rounded py-1 border-grey">
            <div class="input-group-prepend">
                <span class="input-group-text border-0 bg-white">
                    <i class="fa fa-search f-12 text-lightest"></i>
                </span>
            </div>
            <input type="text" id="search-setting-menu" class="form-control border-0 f-14 pl-0"
                   placeholder="@lang('app.search')">
        </div>
    </form>
    <!-- SETTINGS SEARCH END -->

    <!-- SETTINGS MENU START -->
    <ul class="settings-menu" id="settingsMenu">
        @if (user()->permission('manage_superadmin_app_settings') == 'all')
            <x-setting-menu-item :active="$activeMenu" menu="app_settings" :href="route('app-settings.index')"
                             :text="__('app.menu.appSettings')"/>
        @endif
            <x-setting-menu-item :active="$activeMenu" menu="profile_settings"
                             :href="route('superadmin.settings.super-admin-profile.index')"
                             :text="__('app.menu.profileSettings')"/>

        @if (user()->permission('manage_superadmin_notification_settings') == 'all')
        <x-setting-menu-item :active="$activeMenu" menu="notification_settings" :href="route('notifications.index')"
                             :text="__('app.menu.notificationSettings')"/>
        @endif
        @if (user()->permission('manage_superadmin_language_settings') == 'all')
        <x-setting-menu-item :active="$activeMenu" menu="language_settings"
                             :href="route('language-settings.index')"
                             :text="__('app.menu.languageSettings')"/>
        @endif
        @if (user()->permission('manage_superadmin_currency_settings') == 'all')
        <x-setting-menu-item :active="$activeMenu" menu="currency_settings"
                             :href="route('superadmin.settings.global-currency-settings.index')"
                             :text="__('app.menu.currencySettings')"/>
        @endif
        @if (user()->permission('manage_superadmin_payment_settings') == 'all')
        <x-setting-menu-item :active="$activeMenu" menu="payment_gateway_settings"
                             :href="route('superadmin.settings.global-payment-gateway-settings.index')"
                             :text="__('app.menu.paymentGatewayCredential')"/>
        @endif
        @if (user()->permission('manage_superadmin_finance_settings') == 'all')
        <x-setting-menu-item :active="$activeMenu" menu="global_invoice_settings" :href="route('superadmin.settings.global-invoice-settings.index')"
                             :text="__('app.menu.financeSettings')"/>
        @endif
        @if (user()->permission('manage_superadmin_custom_field_settings') == 'all')
        <x-setting-menu-item :active="$activeMenu" menu="custom_fields"
                             :href="route('superadmin.settings.global-custom-fields.index')"
                             :text="__('app.menu.customFields')"/>
        @endif
        @if (user()->permission('manage_superadmin_permission_settings') == 'all')
        <x-setting-menu-item :active="$activeMenu" menu="superadmin_role_permissions"
                             :href="route('superadmin.settings.superadmin-permissions.index')"
                             :text="__('superadmin.superadminRoleAndPermission')"/>
        @endif
        @if (user()->permission('manage_superadmin_storage_settings') == 'all')
        <x-setting-menu-item :active="$activeMenu" menu="storage_settings" :href="route('storage-settings.index')"
                             :text="__('app.menu.storageSettings')"/>
        @endif
        @if (user()->permission('manage_superadmin_social_settings') == 'all')
        <x-setting-menu-item :active="$activeMenu" menu="social_auth_settings"
                             :href="route('social-auth-settings.index')" :text="__('app.menu.socialLogin')"/>
        @endif

        <x-setting-menu-item :active="$activeMenu" menu="security_settings" :href="route('security-settings.index')"
                             :text="__('app.menu.securitySettings')"/>

        @if (user()->permission('manage_superadmin_calendar_settings') == 'all')
            <x-setting-menu-item :active="$activeMenu" menu="google_calendar_settings"
                             :href="route('google-calendar-settings.index')"
                             :text="__('app.menu.googleCalendarSetting')"/>
        @endif
        @if (user()->permission('manage_superadmin_theme_settings') == 'all')
            <x-setting-menu-item :active="$activeMenu" menu="theme_settings"
                             :href="route('superadmin.settings.super-admin-theme-settings.index')"
                             :text="__('app.menu.themeSettings')"/>
        @endif
        @if (user()->permission('manage_superadmin_custom_module_settings') == 'all')
        <x-setting-menu-item :active="$activeMenu" menu="module_settings"
                             :href="route('superadmin.settings.custom-module-settings.index')"
                             :text="__('app.menu.moduleSettings')"/>
        @endif

        @foreach (worksuite_plugins() as $item)
            @includeIf(strtolower($item).'::sections.superadmin.setting-sidebar')
        @endforeach

        @if (user()->permission('manage_superadmin_database_backup_settings') == 'all')
        <x-setting-menu-item :active="$activeMenu" menu="database_backup_settings"
                             :href="route('database-backup-settings.index')"
                             :text="__('app.menu.databaseBackupSetting')"/>
        @endif

        @if (global_setting()->system_update && user()->permission('manage_superadmin_update_settings') == 'all')
            <x-setting-menu-item :active="$activeMenu" menu="update_settings" :href="route('update-settings.index')"
                                 :text="__('app.menu.updates')"/>

        @endif
    </ul>
    <!-- SETTINGS MENU END -->

</div>
<!-- SETTINGS SIDEBAR END -->

<script>
    $("body").on("click", ".ajax-tab", function (event) {
        event.preventDefault();

        $('.project-menu .p-sub-menu').removeClass('active');
        $(this).addClass('active');

        const requestUrl = this.href;

        $.easyAjax({
            url: requestUrl,
            blockUI: true,
            container: ".content-wrapper",
            historyPush: true,
            success: function (response) {
                if (response.status === "success") {
                    $('.content-wrapper').html(response.html);
                    init('.content-wrapper');
                }
            }
        });
    });

    $("#search-setting-menu").on("keyup", function () {
        var value = this.value.toLowerCase().trim();
        $("#settingsMenu li").show().filter(function () {
            return $(this).text().toLowerCase().trim().indexOf(value) == -1;
        }).hide();
    });
</script>
