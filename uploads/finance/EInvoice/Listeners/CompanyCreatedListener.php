<?php

namespace Modules\EInvoice\Listeners;

use Modules\EInvoice\Entities\EInvoiceSetting;

class CompanyCreatedListener
{

    /**
     * Handle the event.
     *
     * @param  object  $event
     * @return void
     */
    public function handle($event)
    {
        $company = $event->company;
        EInvoiceSetting::addModuleSetting($company);
    }

}
