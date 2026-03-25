<?php

namespace Modules\WorkOrders\Listeners;

use Illuminate\Support\Facades\Http;

class SendWorkOrderWebhook
{
    public function handle($event): void
    {
        $url = config('workorders.webhook_url');
        if (!$url) return;

        $payload = [
            'topic' => method_exists($event, 'topic') ? $event->topic() : 'workorders.event',
            'data'  => method_exists($event, 'toArray')
                ? $event->toArray()
                : (property_exists($event, 'workOrder') ? ['work_order_id' => $event->workOrder->id] : []),
            'ts'    => now()->toIso8601String(),
        ];

        try { Http::asJson()->post($url, $payload); } catch (\Throwable $e) { /* log if needed */ }
    }
}
