<?php

namespace Modules\Complaint\Observers;

use Modules\Complaint\Entities\ComplaintType;

class ComplaintTypeObserver
{

    public function creating(ComplaintType $model)
    {
        if (company()) {
            $model->company_id = company()->id;
        }
    }

}
