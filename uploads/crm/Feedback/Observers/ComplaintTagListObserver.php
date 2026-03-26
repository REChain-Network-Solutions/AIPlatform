<?php

namespace Modules\Feedback\Observers;

use Modules\Feedback\Entities\FeedbackTagList;

class FeedbackTagListObserver
{

    public function creating(FeedbackTagList $model)
    {
        if (company()) {
            $model->company_id = company()->id;
        }
    }

}
