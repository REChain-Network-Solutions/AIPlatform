<?php

namespace Modules\Feedback\Observers;

use Modules\Feedback\Entities\FeedbackTag;

class FeedbackTagObserver
{

    public function creating(FeedbackTag $model)
    {
        if (company()) {
            $model->company_id = company()->id;
        }
    }

}
