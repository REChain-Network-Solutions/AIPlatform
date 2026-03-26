<?php

namespace Modules\Feedback\Observers;

use Modules\Feedback\Entities\FeedbackAgentGroups;

class FeedbackAgentGroupsObserver
{

    public function creating(FeedbackAgentGroups $model)
    {
        if (company()) {
            $model->company_id = company()->id;
        }
    }

}
