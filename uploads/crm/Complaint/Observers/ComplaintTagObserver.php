<?php

namespace Modules\Complaint\Observers;

use Modules\Complaint\Entities\ComplaintTag;

class ComplaintTagObserver
{

    public function creating(ComplaintTag $model)
    {
        if (company()) {
            $model->company_id = company()->id;
        }
    }

}
