<?php

namespace Modules\WorkOrders\Listeners;

use Illuminate\Support\Facades\Log;
use Modules\WorkOrders\Events\{WorkOrderCreated, WorkOrderUpdated, WorkOrderCompleted};

class LogWorkOrderActivity
{
    public function handle($event): void
    {
        $type = is_object($event) ? class_basename($event) : 'UnknownEvent';
        try {
            $wo = $event->workOrder ?? null;
            $id = $wo ? ($wo->id ?? null) : null;
            Log::info('[WorkOrders] Activity', ['event'=>$type, 'work_order_id'=>$id]);
        } catch (\Throwable $e) {
            Log::warning('[WorkOrders] Activity log failed: '.$e->getMessage());
        }
    }
}
