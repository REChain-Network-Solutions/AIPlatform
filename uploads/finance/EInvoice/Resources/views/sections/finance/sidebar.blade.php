@if (in_array('invoices', user_modules()) && $sidebarUserPermissions['view_invoices'] != 5 && $sidebarUserPermissions['view_invoices'] != 'none')
    <x-sub-menu-item :text="__('einvoice::app.menu.einvoice')"
                     :link="route('einvoice.index')"
                     :addon="App::environment('demo')"
    />
@endif
