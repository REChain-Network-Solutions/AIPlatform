<?php

namespace Modules\PMCore\app\Console;

use Illuminate\Console\Command;
use Modules\PMCore\app\Models\Timesheet;

class UpdateTimesheetAmounts extends Command
{
    /**
     * The name and signature of the console command.
     */
    protected $signature = 'pmcore:update-timesheet-amounts';

    /**
     * The console command description.
     */
    protected $description = 'Update cost_amount and billable_amount for existing timesheets';

    /**
     * Execute the console command.
     */
    public function handle()
    {
        $this->info('Updating timesheet amounts...');

        $timesheets = Timesheet::all();
        $count = 0;

        foreach ($timesheets as $timesheet) {
            $costAmount = $timesheet->hours * ($timesheet->cost_rate ?: 0);
            $billableAmount = $timesheet->is_billable ? ($timesheet->hours * ($timesheet->billing_rate ?: 0)) : 0;

            $timesheet->update([
                'cost_amount' => $costAmount,
                'billable_amount' => $billableAmount,
            ]);

            $count++;
        }

        $this->info("Updated {$count} timesheets successfully!");

        return Command::SUCCESS;
    }
}
