<?php

namespace Modules\Complaint\Observers;

use Modules\Complaint\Events\ComplaintEvent;
use Modules\Complaint\Events\ComplaintRequesterEvent;
use App\Models\Notification;
use Modules\Complaint\Entities\Complaint;
use Modules\Complaint\Entities\ComplaintEmailSetting;
use App\Models\UniversalSearch;

class ComplaintEmailSettingObserver
{

    public function creating(ComplaintEmailSetting $model)
    {
        if (company()) {
            $model->company_id = company()->id;
        }
    }

}
