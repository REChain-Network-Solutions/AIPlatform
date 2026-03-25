<?php
namespace Modules\WorkOrders\Console\Commands;
use Illuminate\Console\Command;
use Carbon\Carbon;
use Modules\WorkOrders\Entities\WorkOrderRecurrence;
use Modules\WorkOrders\Entities\WorkOrder;
use Modules\WorkOrders\Services\RecurrenceService;
class GenerateRecurringWorkOrders extends Command {
    protected $signature = 'workorders:generate-recurring {--horizon=30}';
    protected $description = 'Generate Work Orders from active recurrence rules up to horizon days';
    public function handle() {
        $horizon = (int)$this->option('horizon');
        $now = Carbon::now();
        $limit = $now->copy()->addDays($horizon);
        $count = 0;
        $recs = WorkOrderRecurrence::where('is_active', true)->get();
        foreach ($recs as $rec) {
            $next = $rec->next_run_at ? Carbon::parse($rec->next_run_at) : Carbon::now();
            while ($next && $next <= $limit) {
                // Duplicate the base WO (simple clone: title, client_id, etc.)
                $base = WorkOrder::find($rec->work_order_id);
                if (!$base) break;
                $new = $base->replicate(['created_at','updated_at','id']);
                $new->status = 'scheduled';
                if ($new->fillable) { /* ignore */ }
                $new->save();
                // update recurrence
                $rec->last_run_at = $next;
                $next = RecurrenceService::nextOccurrence($rec->rrule, $next);
                $rec->next_run_at = $next;
                $rec->save();
                $count++;
            }
        }
        $this->info("Generated {$count} work orders.");
        return 0;
    }
}