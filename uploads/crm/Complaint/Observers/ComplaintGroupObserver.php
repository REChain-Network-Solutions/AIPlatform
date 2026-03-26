<?php

namespace Modules\Complaint\Observers;

use Modules\Complaint\Entities\ComplaintGroup;

class ComplaintGroupObserver
{

    public function creating(ComplaintGroup $model)
    {
        if (company()) {
            $model->company_id = company()->id;
        }
    }

}
