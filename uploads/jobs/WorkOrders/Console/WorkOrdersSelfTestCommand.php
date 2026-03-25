<?php

namespace Modules\WorkOrders\Console;

use Illuminate\Console\Command;
use Illuminate\Support\Facades\Schema;
use Modules\WorkOrders\Jobs\WebhookDispatchJob;

class WorkOrdersSelfTestCommand extends Command
{
    protected $signature = 'workorders:selftest {--queue}';
    protected $description = 'Quick smoke test for Work Orders config, DB, and optional queue dispatch';

    public function handle(): int
    {
        $ok = true;
        $this->info('[WorkOrders] Self test start');

        // Config check
        $auth = config('workorders.api_auth');
        $this->line(' - api_auth: '.var_export($auth, true));

        // DB table check
        $hasFailed = Schema::hasTable('workorders_failed_webhooks');
        $this->line(' - failed_webhooks table: '.($hasFailed ? 'ok' : 'missing'));

        if (!$hasFailed) { $ok = false; }

        if ($this->option('queue')) {
            $this->line(' - dispatching test queue job');
            WebhookDispatchJob::dispatch(['event'=>'SelfTest','ts'=>now()->toISOString()]);
        }

        $this->info($ok ? '[WorkOrders] Self test OK' : '[WorkOrders] Self test found issues');
        return $ok ? 0 : 1;
    }
}
