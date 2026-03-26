<?php

namespace Modules\Complaint\Observers;

use Modules\Complaint\Entities\ComplaintAgentGroups;

class ComplaintAgentGroupsObserver
{

    public function creating(ComplaintAgentGroups $model)
    {
        if (company()) {
            $model->company_id = company()->id;
        }
    }

}
