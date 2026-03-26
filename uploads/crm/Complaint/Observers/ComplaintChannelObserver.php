<?php

namespace Modules\Complaint\Observers;

use Modules\Complaint\Entities\ComplaintChannel;

class ComplaintChannelObserver
{

    public function creating(ComplaintChannel $model)
    {
        if (company()) {
            $model->company_id = company()->id;
        }
    }

}
