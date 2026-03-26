<?php

namespace Modules\Feedback\Observers;

use Modules\Feedback\Entities\FeedbackCustomForm;

class FeedbackCustomFormObserver
{

    public function creating(FeedbackCustomForm $model)
    {
        if (company()) {
            $model->company_id = company()->id;
        }
    }

}
