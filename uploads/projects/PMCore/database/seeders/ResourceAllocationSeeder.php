<?php

namespace Modules\PMCore\database\seeders;

use App\Models\User;
use Carbon\Carbon;
use Illuminate\Database\Seeder;
use Illuminate\Support\Facades\DB;
use Modules\PMCore\app\Models\Project;

class ResourceAllocationSeeder extends Seeder
{
    /**
     * Run the database seeds.
     */
    public function run(): void
    {
        // Get users and projects
        $users = User::whereHas('roles', function ($q) {
            $q->whereNotIn('name', ['client', 'customer']);
        })->limit(15)->get();

        $projects = Project::all();

        if ($users->isEmpty() || $projects->isEmpty()) {
            $this->command->warn('No users or projects found. Please seed users and projects first.');

            return;
        }

        $this->command->info('Seeding resource allocations...');

        $allocations = [];
        $now = now();

        foreach ($projects as $project) {
            // Each project should have 3-6 team members
            $teamSize = rand(3, 6);
            $projectUsers = $users->random(min($teamSize, $users->count()));

            foreach ($projectUsers as $user) {
                // Determine allocation percentage based on role
                $isLead = $projectUsers->first()->id === $user->id;
                $allocationPercentage = $isLead ? rand(80, 100) : rand(40, 80);

                // Hours per day (based on allocation percentage of 8-hour day)
                $hoursPerDay = round(8 * ($allocationPercentage / 100), 1);

                // Determine allocation dates based on project status
                if ($project->status->value === 'completed') {
                    // Completed projects - past allocations
                    $startDate = Carbon::parse($project->start_date);
                    $endDate = Carbon::parse($project->end_date);
                    $status = 'completed';
                } elseif ($project->status->value === 'planning') {
                    // Planning projects - future allocations
                    $startDate = Carbon::parse($project->start_date);
                    $endDate = Carbon::parse($project->end_date);
                    $status = 'planned';
                } else {
                    // Active projects - current allocations
                    $startDate = Carbon::parse($project->start_date);
                    $endDate = null; // Ongoing
                    $status = 'active';
                }

                // Add some variance to dates
                if (rand(1, 10) <= 3 && $startDate->isFuture()) {
                    // 30% chance to start a few days later
                    $startDate->addDays(rand(1, 7));
                }

                $allocations[] = [
                    'project_id' => $project->id,
                    'user_id' => $user->id,
                    'allocation_percentage' => $allocationPercentage,
                    'hours_per_day' => $hoursPerDay,
                    'start_date' => $startDate->format('Y-m-d'),
                    'end_date' => $endDate ? $endDate->format('Y-m-d') : null,
                    'status' => $status,
                    'notes' => $this->getRandomNote($isLead),
                    'is_billable' => rand(1, 10) <= 8, // 80% billable
                    'is_confirmed' => $status !== 'planned',
                    'allocation_type' => 'project',
                    'created_by_id' => 1,
                    'updated_by_id' => 1,
                    'created_at' => $now,
                    'updated_at' => $now,
                ];
            }
        }

        // Insert in chunks
        $chunks = array_chunk($allocations, 50);
        foreach ($chunks as $chunk) {
            DB::table('resource_allocations')->insert($chunk);
        }

        $totalAllocations = count($allocations);
        $this->command->info("Successfully seeded {$totalAllocations} resource allocations!");

        // Show statistics
        $stats = DB::table('resource_allocations')
            ->selectRaw('status, COUNT(*) as count, AVG(allocation_percentage) as avg_allocation')
            ->groupBy('status')
            ->get();

        $this->command->info('Resource Allocation Statistics:');
        foreach ($stats as $stat) {
            $this->command->info("  - {$stat->status}: {$stat->count} allocations, ".round($stat->avg_allocation, 1).'% average allocation');
        }
    }

    /**
     * Get a random role for team members
     */
    private function getRandomRole(): string
    {
        $roles = [
            'Senior Developer',
            'Developer',
            'Junior Developer',
            'UI/UX Designer',
            'QA Engineer',
            'Business Analyst',
            'DevOps Engineer',
            'Technical Writer',
            'Database Administrator',
            'Frontend Developer',
            'Backend Developer',
            'Full Stack Developer',
        ];

        return $roles[array_rand($roles)];
    }

    /**
     * Get a random note for the allocation
     */
    private function getRandomNote($isLead): ?string
    {
        if (rand(1, 10) <= 5) {
            return null; // 50% chance of no note
        }

        $role = $isLead ? 'Lead Developer' : $this->getRandomRole();

        $leadNotes = [
            "Role: {$role}. Primary technical lead for the project",
            "Role: {$role}. Responsible for architecture decisions and code reviews",
            "Role: {$role}. Point of contact for technical discussions",
            "Role: {$role}. Managing development team and sprint planning",
        ];

        $memberNotes = [
            "Role: {$role}. Focusing on frontend development",
            "Role: {$role}. Working on API integration",
            "Role: {$role}. Handling database optimization",
            "Role: {$role}. Responsible for testing and quality assurance",
            "Role: {$role}. Part-time allocation due to other commitments",
            "Role: {$role}. Remote team member",
            "Role: {$role}. Specialized in performance optimization",
            "Role: {$role}. Working on security implementations",
        ];

        return $isLead ? $leadNotes[array_rand($leadNotes)] : $memberNotes[array_rand($memberNotes)];
    }
}
