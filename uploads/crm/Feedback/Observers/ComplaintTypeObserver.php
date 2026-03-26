<?php

namespace Modules\Feedback\Observers;

use Modules\Feedback\Entities\FeedbackType;

class FeedbackTypeObserver
{

    public function creating(FeedbackType $model)
    {
        if (company()) {
            $model->company_id = company()->id;
        }
    }

}
