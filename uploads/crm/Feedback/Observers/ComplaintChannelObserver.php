<?php

namespace Modules\Feedback\Observers;

use Modules\Feedback\Entities\FeedbackChannel;

class FeedbackChannelObserver
{

    public function creating(FeedbackChannel $model)
    {
        if (company()) {
            $model->company_id = company()->id;
        }
    }

}
