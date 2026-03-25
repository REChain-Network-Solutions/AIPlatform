<?php

namespace Modules\WorkOrders\Events;

use Modules\WorkOrders\Entities\WorkOrder;

class WorkOrderCompleted.php
{
    public function __construct(public WorkOrder $workOrder) {}
    public function topic(): string { return 'workorders.completed'; }
}
