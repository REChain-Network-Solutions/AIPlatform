<?php

namespace Modules\PMCore\Database\Seeders;

use Carbon\Carbon;
use Illuminate\Database\Seeder;
use Modules\CRMCore\app\Models\Task;
use Modules\CRMCore\app\Models\TaskPriority;
use Modules\CRMCore\app\Models\TaskStatus;
use Modules\PMCore\app\Models\Project;

class ProjectTaskSeeder extends Seeder
{
    /**
     * Run the database seeds.
     */
    public function run(): void
    {
        $this->command->info('Seeding Project Tasks...');

        // Get all projects to seed tasks for each
        $projects = Project::all();

        if ($projects->isEmpty()) {
            $this->command->info('No projects found. Skipping task seeder.');

            return;
        }

        // Get task statuses and priorities once
        $statuses = [
            'todo' => TaskStatus::where('name', 'To Do')->first(),
            'inProgress' => TaskStatus::where('name', 'In Progress')->first(),
            'waiting' => TaskStatus::where('name', 'Waiting for Input')->first(),
            'completed' => TaskStatus::where('name', 'Completed')->first(),
            'cancelled' => TaskStatus::where('name', 'Cancelled')->first(),
        ];

        $priorities = [
            'low' => TaskPriority::where('name', 'Low')->first(),
            'medium' => TaskPriority::where('name', 'Medium')->first(),
            'high' => TaskPriority::where('name', 'High')->first(),
            'urgent' => TaskPriority::where('name', 'Urgent')->first(),
        ];

        // Process each project
        foreach ($projects as $project) {
            $this->command->info("Creating tasks for project: {$project->name}");
            $this->createTasksForProject($project, $statuses, $priorities);
        }

        $this->command->info('Project tasks seeding completed!');
    }

    /**
     * Create tasks for a specific project based on its status and type
     */
    private function createTasksForProject(Project $project, array $statuses, array $priorities): void
    {
        // Get project members for task assignment
        $projectMembers = $project->members()->with('user')->get();
        $memberIds = $projectMembers->pluck('user.id')->toArray();

        if (empty($memberIds)) {
            // If no project members, use the project manager
            $memberIds = [$project->project_manager_id];
        }

        // Get tasks based on project
        $tasksData = $this->getTasksForProject($project, $statuses, $priorities, $memberIds);

        // Create tasks
        foreach ($tasksData as $taskData) {
            $task = $project->tasks()->create(array_merge($taskData, [
                'created_by_id' => $project->project_manager_id,
                'updated_by_id' => $project->project_manager_id,
            ]));

            $this->command->info("  - Created task: {$task->title}");
        }

        // Create some subtasks for specific projects
        if (in_array($project->code, ['WEB-001', 'MOB-002', 'ECOM-004'])) {
            $this->createSubtasksForProject($project, $statuses, $priorities, $memberIds);
        }
    }

    /**
     * Get tasks based on project characteristics
     */
    private function getTasksForProject(Project $project, array $statuses, array $priorities, array $memberIds): array
    {
        switch ($project->code) {
            case 'WEB-001': // Website Redesign Project
                return $this->getWebsiteRedesignTasks($project, $statuses, $priorities, $memberIds);

            case 'MOB-002': // Mobile App Development
                return $this->getMobileAppTasks($project, $statuses, $priorities, $memberIds);

            case 'INT-003': // Internal Training System
                return $this->getInternalTrainingTasks($project, $statuses, $priorities, $memberIds);

            case 'ECOM-004': // E-commerce Platform (Completed)
                return $this->getCompletedEcommerceTasks($project, $statuses, $priorities, $memberIds);

            case 'API-005': // API Development Project
                return $this->getApiDevelopmentTasks($project, $statuses, $priorities, $memberIds);

            case 'MIG-006': // Legacy System Migration (On Hold)
                return $this->getLegacyMigrationTasks($project, $statuses, $priorities, $memberIds);

            case 'DATA-007': // Data Analytics Dashboard (Completed)
                return $this->getCompletedDataAnalyticsTasks($project, $statuses, $priorities, $memberIds);

            case 'SEC-008': // Security Audit (Cancelled)
                return $this->getCancelledSecurityAuditTasks($project, $statuses, $priorities, $memberIds);

            default:
                return [];
        }
    }

    /**
     * Tasks for Website Redesign Project (In Progress)
     */
    private function getWebsiteRedesignTasks($project, $statuses, $priorities, $memberIds): array
    {
        return [
            // Completed tasks
            [
                'title' => '🎯 Project Kickoff',
                'description' => 'Initial project meeting with stakeholders',
                'task_status_id' => $statuses['completed']->id,
                'task_priority_id' => $priorities['high']->id,
                'assigned_to_user_id' => $project->project_manager_id,
                'due_date' => Carbon::now()->subDays(25),
                'estimated_hours' => 2,
                'actual_hours' => 2.5,
                'is_milestone' => true,
                'completed_at' => Carbon::now()->subDays(25),
                'task_order' => 1,
            ],
            [
                'title' => 'Requirements Gathering',
                'description' => 'Document all functional and non-functional requirements',
                'task_status_id' => $statuses['completed']->id,
                'task_priority_id' => $priorities['high']->id,
                'assigned_to_user_id' => $memberIds[array_rand($memberIds)],
                'due_date' => Carbon::now()->subDays(22),
                'estimated_hours' => 8,
                'actual_hours' => 10,
                'completed_at' => Carbon::now()->subDays(22),
                'task_order' => 2,
            ],
            [
                'title' => 'Wireframe Design',
                'description' => 'Create low-fidelity wireframes for all pages',
                'task_status_id' => $statuses['completed']->id,
                'task_priority_id' => $priorities['medium']->id,
                'assigned_to_user_id' => $memberIds[array_rand($memberIds)],
                'due_date' => Carbon::now()->subDays(18),
                'estimated_hours' => 16,
                'actual_hours' => 14,
                'completed_at' => Carbon::now()->subDays(18),
                'task_order' => 3,
            ],

            // In Progress tasks
            [
                'title' => 'Homepage Development',
                'description' => 'Implement responsive homepage with all sections',
                'task_status_id' => $statuses['inProgress']->id,
                'task_priority_id' => $priorities['high']->id,
                'assigned_to_user_id' => $memberIds[array_rand($memberIds)],
                'due_date' => Carbon::now()->addDays(5),
                'estimated_hours' => 20,
                'time_started_at' => Carbon::now()->subHours(3),
                'task_order' => 4,
            ],
            [
                'title' => 'Navigation Menu',
                'description' => 'Create responsive navigation with dropdown menus',
                'task_status_id' => $statuses['inProgress']->id,
                'task_priority_id' => $priorities['medium']->id,
                'assigned_to_user_id' => $memberIds[array_rand($memberIds)],
                'due_date' => Carbon::now()->addDays(3),
                'estimated_hours' => 8,
                'time_started_at' => Carbon::now()->subDays(1),
                'task_order' => 5,
            ],

            // To Do tasks
            [
                'title' => 'Contact Form Integration',
                'description' => 'Implement contact form with email notifications',
                'task_status_id' => $statuses['todo']->id,
                'task_priority_id' => $priorities['medium']->id,
                'assigned_to_user_id' => $memberIds[array_rand($memberIds)],
                'due_date' => Carbon::now()->addDays(10),
                'estimated_hours' => 6,
                'task_order' => 6,
            ],
            [
                'title' => 'SEO Optimization',
                'description' => 'Implement meta tags, sitemap, and schema markup',
                'task_status_id' => $statuses['todo']->id,
                'task_priority_id' => $priorities['low']->id,
                'assigned_to_user_id' => $memberIds[array_rand($memberIds)],
                'due_date' => Carbon::now()->addDays(20),
                'estimated_hours' => 8,
                'task_order' => 7,
            ],
            [
                'title' => '🎯 Go Live',
                'description' => 'Deploy to production and DNS update',
                'task_status_id' => $statuses['todo']->id,
                'task_priority_id' => $priorities['urgent']->id,
                'assigned_to_user_id' => $project->project_manager_id,
                'due_date' => Carbon::now()->addDays(30),
                'estimated_hours' => 4,
                'is_milestone' => true,
                'task_order' => 8,
            ],
        ];
    }

    /**
     * Tasks for Mobile App Development (Planning)
     */
    private function getMobileAppTasks($project, $statuses, $priorities, $memberIds): array
    {
        return [
            [
                'title' => 'Market Research',
                'description' => 'Analyze competitor apps and market trends',
                'task_status_id' => $statuses['todo']->id,
                'task_priority_id' => $priorities['high']->id,
                'assigned_to_user_id' => $project->project_manager_id,
                'due_date' => Carbon::now()->addDays(20),
                'estimated_hours' => 16,
                'task_order' => 1,
            ],
            [
                'title' => 'User Personas & Journey',
                'description' => 'Define target users and their journey maps',
                'task_status_id' => $statuses['todo']->id,
                'task_priority_id' => $priorities['high']->id,
                'assigned_to_user_id' => $memberIds[array_rand($memberIds)],
                'due_date' => Carbon::now()->addDays(25),
                'estimated_hours' => 12,
                'task_order' => 2,
            ],
            [
                'title' => 'Technical Architecture',
                'description' => 'Design app architecture and technology stack',
                'task_status_id' => $statuses['todo']->id,
                'task_priority_id' => $priorities['medium']->id,
                'assigned_to_user_id' => $memberIds[array_rand($memberIds)],
                'due_date' => Carbon::now()->addDays(30),
                'estimated_hours' => 20,
                'task_order' => 3,
            ],
            [
                'title' => 'UI/UX Design Mockups',
                'description' => 'Create high-fidelity designs for all screens',
                'task_status_id' => $statuses['todo']->id,
                'task_priority_id' => $priorities['medium']->id,
                'assigned_to_user_id' => $memberIds[array_rand($memberIds)],
                'due_date' => Carbon::now()->addDays(40),
                'estimated_hours' => 40,
                'task_order' => 4,
            ],
            [
                'title' => '🎯 Design Approval',
                'description' => 'Get client approval on app designs',
                'task_status_id' => $statuses['todo']->id,
                'task_priority_id' => $priorities['high']->id,
                'assigned_to_user_id' => $project->project_manager_id,
                'due_date' => Carbon::now()->addDays(45),
                'estimated_hours' => 2,
                'is_milestone' => true,
                'task_order' => 5,
            ],
        ];
    }

    /**
     * Tasks for Internal Training System (In Progress)
     */
    private function getInternalTrainingTasks($project, $statuses, $priorities, $memberIds): array
    {
        return [
            // Completed
            [
                'title' => 'Training Needs Assessment',
                'description' => 'Survey employees for training requirements',
                'task_status_id' => $statuses['completed']->id,
                'task_priority_id' => $priorities['medium']->id,
                'assigned_to_user_id' => $project->project_manager_id,
                'due_date' => Carbon::now()->subDays(10),
                'estimated_hours' => 8,
                'actual_hours' => 10,
                'completed_at' => Carbon::now()->subDays(10),
                'task_order' => 1,
            ],

            // In Progress
            [
                'title' => 'Course Structure Design',
                'description' => 'Design course modules and learning paths',
                'task_status_id' => $statuses['inProgress']->id,
                'task_priority_id' => $priorities['high']->id,
                'assigned_to_user_id' => $memberIds[array_rand($memberIds)],
                'due_date' => Carbon::now()->addDays(5),
                'estimated_hours' => 16,
                'time_started_at' => Carbon::now()->subDays(2),
                'task_order' => 2,
            ],

            // To Do
            [
                'title' => 'LMS Platform Setup',
                'description' => 'Install and configure learning management system',
                'task_status_id' => $statuses['todo']->id,
                'task_priority_id' => $priorities['medium']->id,
                'assigned_to_user_id' => $memberIds[array_rand($memberIds)],
                'due_date' => Carbon::now()->addDays(15),
                'estimated_hours' => 12,
                'task_order' => 3,
            ],
            [
                'title' => 'Content Creation',
                'description' => 'Create training videos and documentation',
                'task_status_id' => $statuses['todo']->id,
                'task_priority_id' => $priorities['low']->id,
                'assigned_to_user_id' => $memberIds[array_rand($memberIds)],
                'due_date' => Carbon::now()->addDays(30),
                'estimated_hours' => 40,
                'task_order' => 4,
            ],
        ];
    }

    /**
     * Tasks for Completed E-commerce Platform
     */
    private function getCompletedEcommerceTasks($project, $statuses, $priorities, $memberIds): array
    {
        $completedDate = Carbon::now()->subDays(30);

        return [
            [
                'title' => '🎯 Project Kickoff',
                'description' => 'Initial planning and requirements gathering',
                'task_status_id' => $statuses['completed']->id,
                'task_priority_id' => $priorities['high']->id,
                'assigned_to_user_id' => $project->project_manager_id,
                'due_date' => $completedDate->copy()->subDays(90),
                'estimated_hours' => 4,
                'actual_hours' => 4,
                'is_milestone' => true,
                'completed_at' => $completedDate->copy()->subDays(90),
                'task_order' => 1,
            ],
            [
                'title' => 'Database Design',
                'description' => 'Design database schema for products, orders, users',
                'task_status_id' => $statuses['completed']->id,
                'task_priority_id' => $priorities['high']->id,
                'assigned_to_user_id' => $memberIds[array_rand($memberIds)],
                'due_date' => $completedDate->copy()->subDays(80),
                'estimated_hours' => 16,
                'actual_hours' => 20,
                'completed_at' => $completedDate->copy()->subDays(80),
                'task_order' => 2,
            ],
            [
                'title' => 'Product Catalog',
                'description' => 'Implement product listing, search, and filters',
                'task_status_id' => $statuses['completed']->id,
                'task_priority_id' => $priorities['high']->id,
                'assigned_to_user_id' => $memberIds[array_rand($memberIds)],
                'due_date' => $completedDate->copy()->subDays(60),
                'estimated_hours' => 40,
                'actual_hours' => 45,
                'completed_at' => $completedDate->copy()->subDays(60),
                'task_order' => 3,
            ],
            [
                'title' => 'Shopping Cart',
                'description' => 'Implement cart functionality with session storage',
                'task_status_id' => $statuses['completed']->id,
                'task_priority_id' => $priorities['high']->id,
                'assigned_to_user_id' => $memberIds[array_rand($memberIds)],
                'due_date' => $completedDate->copy()->subDays(45),
                'estimated_hours' => 24,
                'actual_hours' => 28,
                'completed_at' => $completedDate->copy()->subDays(45),
                'task_order' => 4,
            ],
            [
                'title' => 'Payment Integration',
                'description' => 'Integrate Stripe and PayPal payment gateways',
                'task_status_id' => $statuses['completed']->id,
                'task_priority_id' => $priorities['urgent']->id,
                'assigned_to_user_id' => $memberIds[array_rand($memberIds)],
                'due_date' => $completedDate->copy()->subDays(30),
                'estimated_hours' => 32,
                'actual_hours' => 40,
                'completed_at' => $completedDate->copy()->subDays(25),
                'task_order' => 5,
            ],
            [
                'title' => 'Order Management',
                'description' => 'Admin panel for order processing and tracking',
                'task_status_id' => $statuses['completed']->id,
                'task_priority_id' => $priorities['medium']->id,
                'assigned_to_user_id' => $memberIds[array_rand($memberIds)],
                'due_date' => $completedDate->copy()->subDays(15),
                'estimated_hours' => 20,
                'actual_hours' => 18,
                'completed_at' => $completedDate->copy()->subDays(15),
                'task_order' => 6,
            ],
            [
                'title' => '🎯 Launch',
                'description' => 'Production deployment and go-live',
                'task_status_id' => $statuses['completed']->id,
                'task_priority_id' => $priorities['urgent']->id,
                'assigned_to_user_id' => $project->project_manager_id,
                'due_date' => $completedDate,
                'estimated_hours' => 8,
                'actual_hours' => 12,
                'is_milestone' => true,
                'completed_at' => $completedDate,
                'task_order' => 7,
            ],
        ];
    }

    /**
     * Tasks for API Development Project (In Progress, Near Completion)
     */
    private function getApiDevelopmentTasks($project, $statuses, $priorities, $memberIds): array
    {
        return [
            // Completed tasks
            [
                'title' => 'API Architecture Design',
                'description' => 'Design RESTful API structure and endpoints',
                'task_status_id' => $statuses['completed']->id,
                'task_priority_id' => $priorities['high']->id,
                'assigned_to_user_id' => $memberIds[array_rand($memberIds)],
                'due_date' => Carbon::now()->subDays(40),
                'estimated_hours' => 16,
                'actual_hours' => 18,
                'completed_at' => Carbon::now()->subDays(40),
                'task_order' => 1,
            ],
            [
                'title' => 'Authentication System',
                'description' => 'Implement JWT-based authentication',
                'task_status_id' => $statuses['completed']->id,
                'task_priority_id' => $priorities['high']->id,
                'assigned_to_user_id' => $memberIds[array_rand($memberIds)],
                'due_date' => Carbon::now()->subDays(35),
                'estimated_hours' => 12,
                'actual_hours' => 14,
                'completed_at' => Carbon::now()->subDays(35),
                'task_order' => 2,
            ],
            [
                'title' => 'Core Endpoints Development',
                'description' => 'Develop main CRUD endpoints for resources',
                'task_status_id' => $statuses['completed']->id,
                'task_priority_id' => $priorities['high']->id,
                'assigned_to_user_id' => $memberIds[array_rand($memberIds)],
                'due_date' => Carbon::now()->subDays(20),
                'estimated_hours' => 40,
                'actual_hours' => 48,
                'completed_at' => Carbon::now()->subDays(15),
                'task_order' => 3,
            ],

            // In Progress
            [
                'title' => 'API Documentation',
                'description' => 'Write comprehensive API documentation with examples',
                'task_status_id' => $statuses['inProgress']->id,
                'task_priority_id' => $priorities['medium']->id,
                'assigned_to_user_id' => $memberIds[array_rand($memberIds)],
                'due_date' => Carbon::now()->addDays(5),
                'estimated_hours' => 16,
                'time_started_at' => Carbon::now()->subDays(2),
                'task_order' => 4,
            ],

            // To Do
            [
                'title' => 'Rate Limiting',
                'description' => 'Implement API rate limiting and throttling',
                'task_status_id' => $statuses['todo']->id,
                'task_priority_id' => $priorities['medium']->id,
                'assigned_to_user_id' => $memberIds[array_rand($memberIds)],
                'due_date' => Carbon::now()->addDays(10),
                'estimated_hours' => 8,
                'task_order' => 5,
            ],
            [
                'title' => 'Final Testing',
                'description' => 'Complete API testing and performance optimization',
                'task_status_id' => $statuses['todo']->id,
                'task_priority_id' => $priorities['high']->id,
                'assigned_to_user_id' => $memberIds[array_rand($memberIds)],
                'due_date' => Carbon::now()->addDays(15),
                'estimated_hours' => 12,
                'task_order' => 6,
            ],
        ];
    }

    /**
     * Tasks for Legacy System Migration (On Hold)
     */
    private function getLegacyMigrationTasks($project, $statuses, $priorities, $memberIds): array
    {
        return [
            // Completed before hold
            [
                'title' => 'Legacy System Analysis',
                'description' => 'Document existing system architecture and data',
                'task_status_id' => $statuses['completed']->id,
                'task_priority_id' => $priorities['high']->id,
                'assigned_to_user_id' => $memberIds[array_rand($memberIds)],
                'due_date' => Carbon::now()->subDays(50),
                'estimated_hours' => 24,
                'actual_hours' => 32,
                'completed_at' => Carbon::now()->subDays(50),
                'task_order' => 1,
            ],
            [
                'title' => 'Data Migration Strategy',
                'description' => 'Plan data migration approach and mapping',
                'task_status_id' => $statuses['completed']->id,
                'task_priority_id' => $priorities['high']->id,
                'assigned_to_user_id' => $memberIds[array_rand($memberIds)],
                'due_date' => Carbon::now()->subDays(40),
                'estimated_hours' => 16,
                'actual_hours' => 20,
                'completed_at' => Carbon::now()->subDays(40),
                'task_order' => 2,
            ],

            // Waiting status due to hold
            [
                'title' => 'Infrastructure Setup',
                'description' => 'Setup cloud infrastructure for new system',
                'task_status_id' => $statuses['waiting']->id,
                'task_priority_id' => $priorities['medium']->id,
                'assigned_to_user_id' => $memberIds[array_rand($memberIds)],
                'due_date' => Carbon::now()->addDays(30),
                'estimated_hours' => 20,
                'task_order' => 3,
            ],
            [
                'title' => 'Data Migration Scripts',
                'description' => 'Develop scripts for data transformation',
                'task_status_id' => $statuses['waiting']->id,
                'task_priority_id' => $priorities['medium']->id,
                'assigned_to_user_id' => $memberIds[array_rand($memberIds)],
                'due_date' => Carbon::now()->addDays(60),
                'estimated_hours' => 40,
                'task_order' => 4,
            ],
        ];
    }

    /**
     * Tasks for Completed Data Analytics Dashboard
     */
    private function getCompletedDataAnalyticsTasks($project, $statuses, $priorities, $memberIds): array
    {
        $completedDate = Carbon::now()->subDays(15);

        return [
            [
                'title' => 'Requirements Analysis',
                'description' => 'Gather requirements for analytics metrics',
                'task_status_id' => $statuses['completed']->id,
                'task_priority_id' => $priorities['high']->id,
                'assigned_to_user_id' => $project->project_manager_id,
                'due_date' => $completedDate->copy()->subDays(75),
                'estimated_hours' => 12,
                'actual_hours' => 16,
                'completed_at' => $completedDate->copy()->subDays(75),
                'task_order' => 1,
            ],
            [
                'title' => 'Data Source Integration',
                'description' => 'Connect to various data sources and APIs',
                'task_status_id' => $statuses['completed']->id,
                'task_priority_id' => $priorities['high']->id,
                'assigned_to_user_id' => $memberIds[array_rand($memberIds)],
                'due_date' => $completedDate->copy()->subDays(60),
                'estimated_hours' => 24,
                'actual_hours' => 32,
                'completed_at' => $completedDate->copy()->subDays(55),
                'task_order' => 2,
            ],
            [
                'title' => 'Dashboard UI Development',
                'description' => 'Create interactive dashboard with charts',
                'task_status_id' => $statuses['completed']->id,
                'task_priority_id' => $priorities['high']->id,
                'assigned_to_user_id' => $memberIds[array_rand($memberIds)],
                'due_date' => $completedDate->copy()->subDays(30),
                'estimated_hours' => 40,
                'actual_hours' => 50,
                'completed_at' => $completedDate->copy()->subDays(25),
                'task_order' => 3,
            ],
            [
                'title' => 'Real-time Updates',
                'description' => 'Implement WebSocket for live data updates',
                'task_status_id' => $statuses['completed']->id,
                'task_priority_id' => $priorities['medium']->id,
                'assigned_to_user_id' => $memberIds[array_rand($memberIds)],
                'due_date' => $completedDate->copy()->subDays(10),
                'estimated_hours' => 16,
                'actual_hours' => 20,
                'completed_at' => $completedDate->copy()->subDays(5),
                'task_order' => 4,
            ],
            [
                'title' => '🎯 Deployment',
                'description' => 'Deploy to production environment',
                'task_status_id' => $statuses['completed']->id,
                'task_priority_id' => $priorities['urgent']->id,
                'assigned_to_user_id' => $project->project_manager_id,
                'due_date' => $completedDate,
                'estimated_hours' => 4,
                'actual_hours' => 6,
                'is_milestone' => true,
                'completed_at' => $completedDate,
                'task_order' => 5,
            ],
        ];
    }

    /**
     * Tasks for Cancelled Security Audit
     */
    private function getCancelledSecurityAuditTasks($project, $statuses, $priorities, $memberIds): array
    {
        return [
            [
                'title' => 'Initial Security Assessment',
                'description' => 'Preliminary security vulnerability scan',
                'task_status_id' => $statuses['completed']->id,
                'task_priority_id' => $priorities['high']->id,
                'assigned_to_user_id' => $memberIds[array_rand($memberIds)],
                'due_date' => Carbon::now()->subDays(25),
                'estimated_hours' => 8,
                'actual_hours' => 8,
                'completed_at' => Carbon::now()->subDays(25),
                'task_order' => 1,
            ],
            [
                'title' => 'GDPR Compliance Audit',
                'description' => 'Review data handling for GDPR compliance',
                'task_status_id' => $statuses['cancelled']->id,
                'task_priority_id' => $priorities['high']->id,
                'assigned_to_user_id' => $memberIds[array_rand($memberIds)],
                'due_date' => Carbon::now()->subDays(10),
                'estimated_hours' => 16,
                'task_order' => 2,
            ],
            [
                'title' => 'Penetration Testing',
                'description' => 'Conduct thorough penetration testing',
                'task_status_id' => $statuses['cancelled']->id,
                'task_priority_id' => $priorities['medium']->id,
                'assigned_to_user_id' => $memberIds[array_rand($memberIds)],
                'due_date' => Carbon::now(),
                'estimated_hours' => 24,
                'task_order' => 3,
            ],
        ];
    }

    /**
     * Create subtasks for specific projects
     */
    private function createSubtasksForProject($project, $statuses, $priorities, $memberIds): void
    {
        // Get a parent task based on project
        $parentTaskTitle = match ($project->code) {
            'WEB-001' => 'Homepage Development',
            'MOB-002' => 'UI/UX Design Mockups',
            'ECOM-004' => 'Shopping Cart',
            default => null
        };

        if (! $parentTaskTitle) {
            return;
        }

        $parentTask = $project->tasks()->where('title', $parentTaskTitle)->first();
        if (! $parentTask) {
            return;
        }

        $subtasks = match ($project->code) {
            'WEB-001' => [
                [
                    'title' => 'Hero Section',
                    'description' => 'Implement hero banner with carousel',
                    'task_status_id' => $statuses['inProgress']->id,
                    'task_priority_id' => $priorities['high']->id,
                    'assigned_to_user_id' => $memberIds[array_rand($memberIds)],
                    'due_date' => Carbon::now()->addDays(2),
                    'estimated_hours' => 4,
                    'parent_task_id' => $parentTask->id,
                    'task_order' => 1,
                ],
                [
                    'title' => 'Features Grid',
                    'description' => 'Create responsive features showcase section',
                    'task_status_id' => $statuses['todo']->id,
                    'task_priority_id' => $priorities['medium']->id,
                    'assigned_to_user_id' => $memberIds[array_rand($memberIds)],
                    'due_date' => Carbon::now()->addDays(3),
                    'estimated_hours' => 6,
                    'parent_task_id' => $parentTask->id,
                    'task_order' => 2,
                ],
                [
                    'title' => 'Testimonials Carousel',
                    'description' => 'Add customer testimonials section',
                    'task_status_id' => $statuses['todo']->id,
                    'task_priority_id' => $priorities['low']->id,
                    'assigned_to_user_id' => $memberIds[array_rand($memberIds)],
                    'due_date' => Carbon::now()->addDays(4),
                    'estimated_hours' => 3,
                    'parent_task_id' => $parentTask->id,
                    'task_order' => 3,
                ],
            ],
            'MOB-002' => [
                [
                    'title' => 'Login/Signup Screens',
                    'description' => 'Design authentication flow screens',
                    'task_status_id' => $statuses['todo']->id,
                    'task_priority_id' => $priorities['high']->id,
                    'assigned_to_user_id' => $memberIds[array_rand($memberIds)],
                    'due_date' => Carbon::now()->addDays(35),
                    'estimated_hours' => 8,
                    'parent_task_id' => $parentTask->id,
                    'task_order' => 1,
                ],
                [
                    'title' => 'Dashboard Design',
                    'description' => 'Create main dashboard layout',
                    'task_status_id' => $statuses['todo']->id,
                    'task_priority_id' => $priorities['high']->id,
                    'assigned_to_user_id' => $memberIds[array_rand($memberIds)],
                    'due_date' => Carbon::now()->addDays(37),
                    'estimated_hours' => 12,
                    'parent_task_id' => $parentTask->id,
                    'task_order' => 2,
                ],
            ],
            'ECOM-004' => [
                [
                    'title' => 'Add to Cart Functionality',
                    'description' => 'AJAX add to cart with quantity selection',
                    'task_status_id' => $statuses['completed']->id,
                    'task_priority_id' => $priorities['high']->id,
                    'assigned_to_user_id' => $memberIds[array_rand($memberIds)],
                    'due_date' => Carbon::now()->subDays(47),
                    'estimated_hours' => 8,
                    'actual_hours' => 10,
                    'completed_at' => Carbon::now()->subDays(47),
                    'parent_task_id' => $parentTask->id,
                    'task_order' => 1,
                ],
                [
                    'title' => 'Cart Update/Remove',
                    'description' => 'Update quantities and remove items',
                    'task_status_id' => $statuses['completed']->id,
                    'task_priority_id' => $priorities['medium']->id,
                    'assigned_to_user_id' => $memberIds[array_rand($memberIds)],
                    'due_date' => Carbon::now()->subDays(45),
                    'estimated_hours' => 6,
                    'actual_hours' => 6,
                    'completed_at' => Carbon::now()->subDays(45),
                    'parent_task_id' => $parentTask->id,
                    'task_order' => 2,
                ],
            ],
            default => []
        };

        foreach ($subtasks as $subtaskData) {
            $subtask = $project->tasks()->create(array_merge($subtaskData, [
                'created_by_id' => $project->project_manager_id,
                'updated_by_id' => $project->project_manager_id,
            ]));

            $this->command->info("    - Created subtask: {$subtask->title}");
        }
    }
}
