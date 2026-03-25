<?php

namespace Modules\PMCore\database\seeders;

use App\Models\User;
use Illuminate\Database\Seeder;
use Illuminate\Support\Facades\DB;
use Modules\CRMCore\app\Models\Task;
use Modules\PMCore\app\Enums\TimesheetStatus;
use Modules\PMCore\app\Models\Project;

class TimesheetSeeder extends Seeder
{
    /**
     * Run the database seeds.
     */
    public function run(): void
    {
        // Get some users and projects
        $users = User::limit(10)->get();
        $projects = Project::limit(5)->get();

        if ($users->isEmpty() || $projects->isEmpty()) {
            $this->command->warn('No users or projects found. Please seed users and projects first.');

            return;
        }

        $this->command->info('Seeding timesheets...');

        // Define some realistic work descriptions
        $descriptions = [
            'Worked on implementing new features for the dashboard',
            'Fixed bugs reported by QA team',
            'Attended project planning meeting and updated documentation',
            'Code review and refactoring of authentication module',
            'Implemented API endpoints for mobile application',
            'Database optimization and query performance improvements',
            'Created unit tests for new features',
            'Updated user interface based on client feedback',
            'Deployed application to staging environment',
            'Researched and implemented caching strategies',
            'Fixed responsive design issues on mobile devices',
            'Integrated third-party payment gateway',
            'Conducted security audit and fixed vulnerabilities',
            'Optimized image loading and lazy loading implementation',
            'Wrote technical documentation for API endpoints',
        ];

        $timesheets = [];
        $now = now();

        foreach ($projects as $project) {
            // Get tasks for this project if available
            // Note: CRM tasks might not have project_id column, so we'll skip task assignment
            $projectTasks = [];

            // Create timesheets for the last 30 days
            for ($daysAgo = 30; $daysAgo >= 0; $daysAgo--) {
                $date = $now->copy()->subDays($daysAgo);

                // Skip weekends for more realistic data
                if ($date->isWeekend()) {
                    continue;
                }

                // Create 2-4 timesheets per day for different users
                $dailyEntries = rand(2, 4);
                $usedUsers = [];

                for ($i = 0; $i < $dailyEntries; $i++) {
                    // Get a random user that hasn't logged time today
                    $user = $users->filter(function ($u) use ($usedUsers) {
                        return ! in_array($u->id, $usedUsers);
                    })->random();

                    if (! $user) {
                        continue;
                    }

                    $usedUsers[] = $user->id;

                    // Random hours between 1 and 8
                    $hours = rand(1, 8) + (rand(0, 1) ? 0.5 : 0);

                    // 80% chance of being billable
                    $isBillable = rand(1, 10) <= 8;

                    // Random billing rate between 50-150
                    $billingRate = $isBillable ? rand(50, 150) : null;

                    // Cost rate is usually 40-60% of billing rate
                    $costRate = $billingRate ? round($billingRate * (rand(40, 60) / 100), 2) : null;

                    // Determine status based on how old the entry is
                    if ($daysAgo > 7) {
                        // Older entries are mostly approved
                        $status = rand(1, 10) <= 8 ? TimesheetStatus::APPROVED : TimesheetStatus::INVOICED;
                    } elseif ($daysAgo > 2) {
                        // Recent entries are submitted or approved
                        $status = rand(1, 2) == 1 ? TimesheetStatus::SUBMITTED : TimesheetStatus::APPROVED;
                    } else {
                        // Very recent entries might be draft
                        $statusRand = rand(1, 3);
                        $status = $statusRand == 1 ? TimesheetStatus::DRAFT :
                                 ($statusRand == 2 ? TimesheetStatus::SUBMITTED : TimesheetStatus::APPROVED);
                    }

                    // Set approval data for approved/rejected/invoiced entries
                    $approvedById = null;
                    $approvedAt = null;
                    if (in_array($status, [TimesheetStatus::APPROVED, TimesheetStatus::REJECTED, TimesheetStatus::INVOICED])) {
                        // Get a different user as approver (preferably a manager)
                        $approver = $users->where('id', '!=', $user->id)->random();
                        $approvedById = $approver->id;
                        $approvedAt = $date->copy()->addDay()->setTime(rand(9, 18), rand(0, 59));
                    }

                    // Calculate cost and billable amounts
                    $costAmount = $hours * ($costRate ?: 0);
                    $billableAmount = $isBillable ? $hours * ($billingRate ?: 0) : 0;

                    $timesheets[] = [
                        'user_id' => $user->id,
                        'project_id' => $project->id,
                        'task_id' => ! empty($projectTasks) && rand(1, 10) <= 7 ? $projectTasks[array_rand($projectTasks)] : null,
                        'date' => $date->format('Y-m-d'),
                        'hours' => $hours,
                        'description' => $descriptions[array_rand($descriptions)],
                        'is_billable' => $isBillable,
                        'billing_rate' => $billingRate,
                        'cost_rate' => $costRate,
                        'cost_amount' => $costAmount,
                        'billable_amount' => $billableAmount,
                        'status' => $status->value,
                        'approved_by_id' => $approvedById,
                        'approved_at' => $approvedAt,
                        'created_by_id' => $user->id,
                        'updated_by_id' => $user->id,
                        'created_at' => $date->copy()->setTime(rand(18, 20), rand(0, 59)),
                        'updated_at' => $date->copy()->setTime(rand(18, 20), rand(0, 59)),
                    ];
                }
            }
        }

        // Insert in chunks for better performance
        $chunks = array_chunk($timesheets, 100);
        foreach ($chunks as $chunk) {
            DB::table('timesheets')->insert($chunk);
        }

        $totalTimesheets = count($timesheets);
        $this->command->info("Successfully seeded {$totalTimesheets} timesheet entries!");

        // Show statistics
        $stats = DB::table('timesheets')
            ->selectRaw('status, COUNT(*) as count, SUM(hours) as total_hours, SUM(cost_amount) as total_cost, SUM(billable_amount) as total_revenue')
            ->groupBy('status')
            ->get();

        $this->command->info('Timesheet Statistics:');
        foreach ($stats as $stat) {
            $this->command->info("  - {$stat->status}: {$stat->count} entries, {$stat->total_hours} hours, Cost: $".number_format($stat->total_cost, 2).', Revenue: $'.number_format($stat->total_revenue, 2));
        }

        // Overall statistics
        $overall = DB::table('timesheets')
            ->selectRaw('COUNT(*) as total_entries, SUM(hours) as total_hours, SUM(cost_amount) as total_cost, SUM(billable_amount) as total_revenue, SUM(CASE WHEN is_billable = 1 THEN hours ELSE 0 END) as billable_hours')
            ->first();

        $this->command->info("\nOverall Statistics:");
        $this->command->info("  - Total Entries: {$overall->total_entries}");
        $this->command->info("  - Total Hours: {$overall->total_hours}");
        $this->command->info("  - Billable Hours: {$overall->billable_hours}");
        $this->command->info('  - Total Cost: $'.number_format($overall->total_cost, 2));
        $this->command->info('  - Total Revenue: $'.number_format($overall->total_revenue, 2));
        $this->command->info('  - Profit Margin: $'.number_format($overall->total_revenue - $overall->total_cost, 2));
    }
}
