<?php

namespace Modules\Complaint\Observers;

use Modules\Complaint\Entities\ComplaintTagList;

class ComplaintTagListObserver
{

    public function creating(ComplaintTagList $model)
    {
        if (company()) {
            $model->company_id = company()->id;
        }
    }

}
