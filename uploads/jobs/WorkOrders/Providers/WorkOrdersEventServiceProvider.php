<?php

namespace Modules\WorkOrders\Providers;

use Illuminate\Foundation\Support\Providers\EventServiceProvider as ServiceProvider;
use Modules\WorkOrders\Events\{WorkOrderCreated, WorkOrderUpdated, WorkOrderCompleted};
use Modules\WorkOrders\Listeners\SendWorkOrderWebhook;
use Modules\WorkOrders\Listeners\AutoConvertOnCompletion;

class WorkOrdersEventServiceProvider extends ServiceProvider
{
    protected $listen = [
        WorkOrderCreated::class => [SendWorkOrderWebhook::class, \Modules\WorkOrders\Listeners\LogWorkOrderActivity::class],
        WorkOrderUpdated::class => [SendWorkOrderWebhook::class, \Modules\WorkOrders\Listeners\LogWorkOrderActivity::class],
        WorkOrderCompleted::class => [SendWorkOrderWebhook::class, AutoConvertOnCompletion::class, \Modules\WorkOrders\Listeners\LogWorkOrderActivity::class],
    ];
}
