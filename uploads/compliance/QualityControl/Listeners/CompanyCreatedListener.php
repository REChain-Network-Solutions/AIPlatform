<?php

namespace Modules\QualityControl\Listeners;

use Modules\QualityControl\Entities\RecurringSchedule;

class CompanyCreatedListener
{

    public function handle($event)
    {
        $company = $event->company;
        RecurringSchedule::addModuleSetting($company);
    }

}
