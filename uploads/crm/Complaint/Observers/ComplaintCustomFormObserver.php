<?php

namespace Modules\Complaint\Observers;

use Modules\Complaint\Entities\ComplaintCustomForm;

class ComplaintCustomFormObserver
{

    public function creating(ComplaintCustomForm $model)
    {
        if (company()) {
            $model->company_id = company()->id;
        }
    }

}
