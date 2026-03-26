<?php

namespace Modules\Feedback\Observers;

use Modules\Feedback\Entities\FeedbackGroup;

class FeedbackGroupObserver
{

    public function creating(FeedbackGroup $model)
    {
        if (company()) {
            $model->company_id = company()->id;
        }
    }

}
