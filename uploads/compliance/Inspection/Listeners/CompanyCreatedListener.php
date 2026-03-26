<?php

namespace Modules\Inspection\Listeners;

use Modules\Inspection\Entities\RecurringSchedule;

class CompanyCreatedListener
{

    public function handle($event)
    {
        $company = $event->company;
        RecurringSchedule::addModuleSetting($company);
    }

}
