<?php

namespace Modules\WorkOrders\Listeners;

use Modules\WorkOrders\Events\WorkOrderCompleted;
use Modules\WorkOrders\Entities\WorkOrdersSetting;

class AutoConvertOnCompletion
{
    public function handle(WorkOrderCompleted $event): void
    {
        $settings = WorkOrdersSetting::getOrCreate();
        if ($settings->auto_convert_on_complete) {
            $event->workOrder->loadMissing('tasks');
            $event->workOrder->convertToProject();
        }
    }
}
