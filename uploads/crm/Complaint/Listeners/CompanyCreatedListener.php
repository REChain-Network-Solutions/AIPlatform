<?php

namespace Modules\Complaint\Listeners;
use Modules\Complaint\Entities\Complaint;

class CompanyCreatedListener
{

    public function handle($event)
    {
        $company = $event->company;
        Complaint::addModuleSetting($company);
    }

}
