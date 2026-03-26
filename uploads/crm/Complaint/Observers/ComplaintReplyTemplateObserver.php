<?php

namespace Modules\Complaint\Observers;

use Modules\Complaint\Entities\ComplaintReplyTemplate;

class ComplaintReplyTemplateObserver
{

    public function creating(ComplaintReplyTemplate $model)
    {
        if (company()) {
            $model->company_id = company()->id;
        }
    }

}
