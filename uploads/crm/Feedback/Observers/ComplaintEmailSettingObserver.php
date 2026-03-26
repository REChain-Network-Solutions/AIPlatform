<?php

namespace Modules\Feedback\Observers;

use Modules\Feedback\Events\FeedbackEvent;
use Modules\Feedback\Events\FeedbackRequesterEvent;
use App\Models\Notification;
use Modules\Feedback\Entities\Feedback;
use Modules\Feedback\Entities\FeedbackEmailSetting;
use App\Models\UniversalSearch;

class FeedbackEmailSettingObserver
{

    public function creating(FeedbackEmailSetting $model)
    {
        if (company()) {
            $model->company_id = company()->id;
        }
    }

}
