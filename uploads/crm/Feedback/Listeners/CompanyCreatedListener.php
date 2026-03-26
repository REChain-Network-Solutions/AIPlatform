<?php

namespace Modules\Feedback\Listeners;
use Modules\Feedback\Entities\Feedback;

class CompanyCreatedListener
{

    public function handle($event)
    {
        $company = $event->company;
        Feedback::addModuleSetting($company);
    }

}
