<?php

namespace Modules\WorkOrders\Events;

use Modules\WorkOrders\Entities\WorkOrder;

class WorkOrderCreated.php
{
    public function __construct(public WorkOrder $workOrder) {}
    public function topic(): string { return 'workorders.created'; }
}
