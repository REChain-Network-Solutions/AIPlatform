<?php

namespace Modules\Feedback\Observers;

use Modules\Feedback\Entities\FeedbackReplyTemplate;

class FeedbackReplyTemplateObserver
{

    public function creating(FeedbackReplyTemplate $model)
    {
        if (company()) {
            $model->company_id = company()->id;
        }
    }

}
