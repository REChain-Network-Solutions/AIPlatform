<?php

namespace Modules\PMCore\Database\Seeders;

use App\Models\User;
use Illuminate\Database\Seeder;
use Modules\PMCore\app\Enums\ProjectPriority;
use Modules\PMCore\app\Enums\ProjectStatus;
use Modules\PMCore\app\Enums\ProjectType;
use Modules\PMCore\app\Models\Project;

class ProjectSeeder extends Seeder
{
    /**
     * Run the database seeds.
     */
    public function run(): void
    {
        // Get first user as project manager
        $user = User::first();

        if (! $user) {
            $this->command->info('No users found. Skipping project seeder.');

            return;
        }

        $projects = [
            [
                'name' => 'Website Redesign Project',
                'code' => 'WEB-001',
                'description' => 'Complete redesign of the company website with modern UI/UX.',
                'status' => ProjectStatus::IN_PROGRESS,
                'type' => ProjectType::CLIENT,
                'priority' => ProjectPriority::HIGH,
                'start_date' => now()->subDays(30),
                'end_date' => now()->addDays(30),
                'budget' => 50000.00,
                'actual_cost' => 22000.00, // In progress, under budget
                'actual_revenue' => 25000.00,
                'completion_percentage' => 45,
                'hourly_rate' => 100.00,
                'color_code' => '#007bff',
                'is_billable' => true,
                'project_manager_id' => $user->id,
            ],
            [
                'name' => 'Mobile App Development',
                'code' => 'MOB-002',
                'description' => 'Native mobile application for iOS and Android platforms.',
                'status' => ProjectStatus::PLANNING,
                'type' => ProjectType::CLIENT,
                'priority' => ProjectPriority::MEDIUM,
                'start_date' => now()->addDays(15),
                'end_date' => now()->addDays(90),
                'budget' => 75000.00,
                'actual_cost' => 0, // Planning phase, no costs yet
                'actual_revenue' => 0,
                'completion_percentage' => 0,
                'hourly_rate' => 120.00,
                'color_code' => '#28a745',
                'is_billable' => true,
                'project_manager_id' => $user->id,
            ],
            [
                'name' => 'Internal Training System',
                'code' => 'INT-003',
                'description' => 'Internal employee training and onboarding system.',
                'status' => ProjectStatus::IN_PROGRESS,
                'type' => ProjectType::INTERNAL,
                'priority' => ProjectPriority::LOW,
                'start_date' => now()->subDays(15),
                'end_date' => now()->addDays(45),
                'budget' => 25000.00,
                'actual_cost' => 8500.00, // Internal project, no revenue
                'actual_revenue' => 0,
                'completion_percentage' => 35,
                'hourly_rate' => 80.00,
                'color_code' => '#ffc107',
                'is_billable' => false,
                'project_manager_id' => $user->id,
            ],
            [
                'name' => 'E-commerce Platform',
                'code' => 'ECOM-004',
                'description' => 'Full-featured e-commerce platform with payment integration.',
                'status' => ProjectStatus::COMPLETED,
                'type' => ProjectType::CLIENT,
                'priority' => ProjectPriority::HIGH,
                'start_date' => now()->subDays(120),
                'end_date' => now()->subDays(30),
                'budget' => 100000.00,
                'actual_cost' => 95000.00, // Completed, slightly under budget
                'actual_revenue' => 120000.00, // Profitable project
                'completion_percentage' => 100,
                'completed_at' => now()->subDays(30),
                'hourly_rate' => 150.00,
                'color_code' => '#dc3545',
                'is_billable' => true,
                'project_manager_id' => $user->id,
            ],
            [
                'name' => 'API Development Project',
                'code' => 'API-005',
                'description' => 'RESTful API development for third-party integrations.',
                'status' => ProjectStatus::IN_PROGRESS,
                'type' => ProjectType::CLIENT,
                'priority' => ProjectPriority::HIGH,
                'start_date' => now()->subDays(45),
                'end_date' => now()->addDays(15),
                'budget' => 40000.00,
                'actual_cost' => 38000.00, // Near completion, over budget
                'actual_revenue' => 35000.00,
                'completion_percentage' => 85,
                'hourly_rate' => 110.00,
                'color_code' => '#6f42c1',
                'is_billable' => true,
                'project_manager_id' => $user->id,
            ],
            [
                'name' => 'Legacy System Migration',
                'code' => 'MIG-006',
                'description' => 'Migration from legacy system to modern cloud-based architecture.',
                'status' => ProjectStatus::ON_HOLD,
                'type' => ProjectType::CLIENT,
                'priority' => ProjectPriority::MEDIUM,
                'start_date' => now()->subDays(60),
                'end_date' => now()->addDays(120),
                'budget' => 150000.00,
                'actual_cost' => 45000.00, // On hold mid-way
                'actual_revenue' => 50000.00,
                'completion_percentage' => 30,
                'hourly_rate' => 130.00,
                'color_code' => '#fd7e14',
                'is_billable' => true,
                'project_manager_id' => $user->id,
            ],
            [
                'name' => 'Data Analytics Dashboard',
                'code' => 'DATA-007',
                'description' => 'Business intelligence dashboard with real-time analytics.',
                'status' => ProjectStatus::COMPLETED,
                'type' => ProjectType::CLIENT,
                'priority' => ProjectPriority::MEDIUM,
                'start_date' => now()->subDays(90),
                'end_date' => now()->subDays(15),
                'budget' => 60000.00,
                'actual_cost' => 72000.00, // Over budget project
                'actual_revenue' => 65000.00, // Loss-making project
                'completion_percentage' => 100,
                'completed_at' => now()->subDays(15),
                'hourly_rate' => 125.00,
                'color_code' => '#20c997',
                'is_billable' => true,
                'project_manager_id' => $user->id,
            ],
            [
                'name' => 'Security Audit & Compliance',
                'code' => 'SEC-008',
                'description' => 'Comprehensive security audit and GDPR compliance implementation.',
                'status' => ProjectStatus::CANCELLED,
                'type' => ProjectType::INTERNAL,
                'priority' => ProjectPriority::HIGH,
                'start_date' => now()->subDays(30),
                'end_date' => now()->addDays(30),
                'budget' => 30000.00,
                'actual_cost' => 5000.00, // Cancelled early
                'actual_revenue' => 0,
                'completion_percentage' => 15,
                'hourly_rate' => 140.00,
                'color_code' => '#e83e8c',
                'is_billable' => false,
                'project_manager_id' => $user->id,
            ],
        ];

        foreach ($projects as $projectData) {
            $project = Project::create($projectData);

            // Add project manager as member
            $project->addMember($user->id, [
                'role' => 'manager',
                'hourly_rate' => $projectData['hourly_rate'],
            ]);
        }

        $this->command->info('Created '.count($projects).' sample projects.');
    }
}
