<?php

namespace Modules\PMCore\Database\Seeders;

use Illuminate\Database\Seeder;
use Spatie\Permission\Models\Permission;
use Spatie\Permission\Models\Role;

class PMCorePermissionSeeder extends Seeder
{
    /**
     * All PMCore permissions organized by category
     */
    protected array $permissions = [];

    /**
     * Run the database seeds.
     */
    public function run(): void
    {
        // Reset cached roles and permissions
        app()[\Spatie\Permission\PermissionRegistrar::class]->forgetCachedPermissions();

        // Define all permissions
        $this->definePermissions();

        // Create permissions
        $this->createPermissions();

        // Update roles
        $this->updateRoles();

        $this->command->info('PMCore permissions seeded successfully!');
    }

    /**
     * Define all PMCore module permissions
     */
    protected function definePermissions(): void
    {
        $this->permissions = [
            // Project Management
            ['name' => 'pmcore.view-projects', 'description' => 'View all projects'],
            ['name' => 'pmcore.view-own-projects', 'description' => 'View own projects only'],
            ['name' => 'pmcore.create-project', 'description' => 'Create new projects'],
            ['name' => 'pmcore.edit-project', 'description' => 'Edit projects'],
            ['name' => 'pmcore.edit-own-project', 'description' => 'Edit own projects only'],
            ['name' => 'pmcore.delete-project', 'description' => 'Delete projects'],
            ['name' => 'pmcore.archive-project', 'description' => 'Archive projects'],
            ['name' => 'pmcore.duplicate-project', 'description' => 'Duplicate projects'],
            ['name' => 'pmcore.export-projects', 'description' => 'Export projects data'],

            // Project Team Management
            ['name' => 'pmcore.manage-project-team', 'description' => 'Manage project team members'],
            ['name' => 'pmcore.view-project-members', 'description' => 'View project team members'],
            ['name' => 'pmcore.add-project-member', 'description' => 'Add team members to projects'],
            ['name' => 'pmcore.remove-project-member', 'description' => 'Remove team members from projects'],
            ['name' => 'pmcore.update-member-role', 'description' => 'Update team member roles'],

            // Project Tasks
            ['name' => 'pmcore.view-project-tasks', 'description' => 'View project tasks'],
            ['name' => 'pmcore.create-project-task', 'description' => 'Create project tasks'],
            ['name' => 'pmcore.edit-project-task', 'description' => 'Edit project tasks'],
            ['name' => 'pmcore.edit-own-task', 'description' => 'Edit own tasks only'],
            ['name' => 'pmcore.delete-project-task', 'description' => 'Delete project tasks'],
            ['name' => 'pmcore.assign-project-task', 'description' => 'Assign project tasks to team members'],
            ['name' => 'pmcore.complete-project-task', 'description' => 'Mark project tasks as complete'],
            ['name' => 'pmcore.start-project-task', 'description' => 'Start working on project tasks'],
            ['name' => 'pmcore.reorder-project-tasks', 'description' => 'Reorder project tasks'],

            // Project Dashboard & Reports
            ['name' => 'pmcore.view-project-dashboard', 'description' => 'View project dashboard'],
            ['name' => 'pmcore.view-project-reports', 'description' => 'View project reports'],
            ['name' => 'pmcore.view-time-reports', 'description' => 'View time tracking reports'],
            ['name' => 'pmcore.view-budget-reports', 'description' => 'View budget reports'],
            ['name' => 'pmcore.view-resource-reports', 'description' => 'View resource utilization reports'],
            ['name' => 'pmcore.export-reports', 'description' => 'Export project reports'],

            // Timesheets
            ['name' => 'pmcore.view-timesheets', 'description' => 'View all timesheets'],
            ['name' => 'pmcore.view-own-timesheets', 'description' => 'View own timesheets only'],
            ['name' => 'pmcore.create-timesheet', 'description' => 'Create timesheets'],
            ['name' => 'pmcore.edit-timesheet', 'description' => 'Edit timesheets'],
            ['name' => 'pmcore.edit-own-timesheet', 'description' => 'Edit own timesheets only'],
            ['name' => 'pmcore.delete-timesheet', 'description' => 'Delete timesheets'],
            ['name' => 'pmcore.submit-timesheet', 'description' => 'Submit timesheets for approval'],
            ['name' => 'pmcore.approve-timesheet', 'description' => 'Approve timesheets'],
            ['name' => 'pmcore.reject-timesheet', 'description' => 'Reject timesheets'],
            ['name' => 'pmcore.export-timesheets', 'description' => 'Export timesheet data'],

            // Resource Management
            ['name' => 'pmcore.view-resources', 'description' => 'View resource allocations'],
            ['name' => 'pmcore.manage-resources', 'description' => 'Manage resource allocations'],
            ['name' => 'pmcore.allocate-resource', 'description' => 'Allocate resources to projects'],
            ['name' => 'pmcore.view-resource-capacity', 'description' => 'View resource capacity planning'],
            ['name' => 'pmcore.view-resource-schedule', 'description' => 'View resource schedules'],
            ['name' => 'pmcore.check-resource-availability', 'description' => 'Check resource availability'],

            // Project Status Management
            ['name' => 'pmcore.view-project-statuses', 'description' => 'View project statuses'],
            ['name' => 'pmcore.manage-project-statuses', 'description' => 'Manage project statuses'],
            ['name' => 'pmcore.create-project-status', 'description' => 'Create new project statuses'],
            ['name' => 'pmcore.edit-project-status', 'description' => 'Edit project statuses'],
            ['name' => 'pmcore.delete-project-status', 'description' => 'Delete project statuses'],
            ['name' => 'pmcore.toggle-project-status', 'description' => 'Enable/disable project statuses'],
            ['name' => 'pmcore.sort-project-statuses', 'description' => 'Reorder project statuses'],

            // Project Settings
            ['name' => 'pmcore.manage-project-settings', 'description' => 'Manage project module settings'],
            ['name' => 'pmcore.view-project-settings', 'description' => 'View project module settings'],
        ];
    }

    /**
     * Create all permissions in the database
     */
    protected function createPermissions(): void
    {
        foreach ($this->permissions as $permission) {
            try {
                $perm = Permission::firstOrCreate(
                    [
                        'name' => $permission['name'],
                        'guard_name' => 'web',
                    ],
                    [
                        'module' => 'PMCore',
                        'description' => $permission['description'],
                    ]
                );

                if ($perm->wasRecentlyCreated) {
                    $this->command->info("Created permission: {$permission['name']}");
                } else {
                    // Update the module and description if permission already exists
                    $perm->update([
                        'module' => 'PMCore',
                        'description' => $permission['description'],
                    ]);
                    $this->command->info("Updated permission: {$permission['name']}");
                }
            } catch (\Exception $e) {
                $this->command->error("Failed to create/update permission {$permission['name']}: ".$e->getMessage());
            }
        }
    }

    /**
     * Update existing roles with PMCore permissions
     */
    protected function updateRoles(): void
    {
        // Super Admin - Full access
        $this->updateSuperAdminRole();

        // Admin - Full project management
        $this->updateAdminRole();

        // Project Manager - Full project access
        $this->updateProjectManagerRole();

        // Team Leader - Limited management
        $this->updateTeamLeaderRole();

        // Employee - Basic access
        $this->updateEmployeeRole();

        // Client - View only access
        $this->updateClientRole();
    }

    /**
     * Update Super Admin role
     */
    protected function updateSuperAdminRole(): void
    {
        try {
            $role = Role::where('name', 'super_admin')->first();
            if ($role) {
                $permissions = Permission::where('name', 'like', 'pmcore.%')->pluck('name')->toArray();
                if (! empty($permissions)) {
                    $role->givePermissionTo($permissions);
                    $this->command->info('Updated Super Admin role with all PMCore permissions ('.count($permissions).' permissions)');
                }
            }
        } catch (\Exception $e) {
            $this->command->error('Failed to update Super Admin role: '.$e->getMessage());
        }
    }

    /**
     * Update Admin role
     */
    protected function updateAdminRole(): void
    {
        try {
            $role = Role::where('name', 'admin')->first();
            if ($role) {
                $permissions = Permission::where('name', 'like', 'pmcore.%')->pluck('name')->toArray();
                $role->givePermissionTo($permissions);
                $this->command->info('Updated Admin role with PMCore permissions');
            }
        } catch (\Exception $e) {
            $this->command->error('Failed to update Admin role: '.$e->getMessage());
        }
    }

    /**
     * Update Project Manager role
     */
    protected function updateProjectManagerRole(): void
    {
        try {
            $role = Role::where('name', 'project_manager')->first();
            if ($role) {
                $permissions = [
                    // Full project access
                    'pmcore.view-projects',
                    'pmcore.create-project',
                    'pmcore.edit-project',
                    'pmcore.delete-project',
                    'pmcore.archive-project',
                    'pmcore.duplicate-project',
                    'pmcore.export-projects',

                    // Team management
                    'pmcore.manage-project-team',
                    'pmcore.view-project-members',
                    'pmcore.add-project-member',
                    'pmcore.remove-project-member',
                    'pmcore.update-member-role',

                    // Tasks
                    'pmcore.view-project-tasks',
                    'pmcore.create-project-task',
                    'pmcore.edit-project-task',
                    'pmcore.delete-project-task',
                    'pmcore.assign-project-task',
                    'pmcore.complete-project-task',
                    'pmcore.start-project-task',
                    'pmcore.reorder-project-tasks',

                    // Dashboard & Reports
                    'pmcore.view-project-dashboard',
                    'pmcore.view-project-reports',
                    'pmcore.view-time-reports',
                    'pmcore.view-budget-reports',
                    'pmcore.view-resource-reports',
                    'pmcore.export-reports',

                    // Timesheets
                    'pmcore.view-timesheets',
                    'pmcore.approve-timesheet',
                    'pmcore.reject-timesheet',
                    'pmcore.export-timesheets',

                    // Resources
                    'pmcore.view-resources',
                    'pmcore.manage-resources',
                    'pmcore.allocate-resource',
                    'pmcore.view-resource-capacity',
                    'pmcore.view-resource-schedule',
                    'pmcore.check-resource-availability',

                    // Status management
                    'pmcore.view-project-statuses',
                ];
                $role->givePermissionTo($permissions);
                $this->command->info('Updated Project Manager role with PMCore permissions');
            }
        } catch (\Exception $e) {
            $this->command->error('Failed to update Project Manager role: '.$e->getMessage());
        }
    }

    /**
     * Update Team Leader role
     */
    protected function updateTeamLeaderRole(): void
    {
        try {
            $role = Role::where('name', 'team_leader')->first();
            if ($role) {
                $permissions = [
                    // View projects
                    'pmcore.view-projects',
                    'pmcore.edit-own-project',

                    // Team viewing
                    'pmcore.view-project-members',

                    // Tasks
                    'pmcore.view-project-tasks',
                    'pmcore.create-project-task',
                    'pmcore.edit-project-task',
                    'pmcore.assign-project-task',
                    'pmcore.complete-project-task',
                    'pmcore.start-project-task',

                    // Dashboard & Reports
                    'pmcore.view-project-dashboard',
                    'pmcore.view-project-reports',
                    'pmcore.view-time-reports',
                    'pmcore.view-resource-reports',

                    // Timesheets
                    'pmcore.view-timesheets',
                    'pmcore.create-timesheet',
                    'pmcore.edit-own-timesheet',
                    'pmcore.submit-timesheet',

                    // Resources
                    'pmcore.view-resources',
                    'pmcore.view-resource-capacity',
                    'pmcore.view-resource-schedule',
                ];
                $role->givePermissionTo($permissions);
                $this->command->info('Updated Team Leader role with PMCore permissions');
            }
        } catch (\Exception $e) {
            $this->command->error('Failed to update Team Leader role: '.$e->getMessage());
        }
    }

    /**
     * Update Employee role
     */
    protected function updateEmployeeRole(): void
    {
        try {
            $role = Role::where('name', 'employee')->first();
            if ($role) {
                $permissions = [
                    // View own projects
                    'pmcore.view-own-projects',

                    // View team
                    'pmcore.view-project-members',

                    // Tasks
                    'pmcore.view-project-tasks',
                    'pmcore.edit-own-task',
                    'pmcore.complete-project-task',
                    'pmcore.start-project-task',

                    // Timesheets
                    'pmcore.view-own-timesheets',
                    'pmcore.create-timesheet',
                    'pmcore.edit-own-timesheet',
                    'pmcore.submit-timesheet',

                    // View schedules
                    'pmcore.view-resource-schedule',
                ];
                $role->givePermissionTo($permissions);
                $this->command->info('Updated Employee role with PMCore permissions');
            }
        } catch (\Exception $e) {
            $this->command->error('Failed to update Employee role: '.$e->getMessage());
        }
    }

    /**
     * Update Client role
     */
    protected function updateClientRole(): void
    {
        try {
            $role = Role::where('name', 'client')->first();
            if ($role) {
                $permissions = [
                    'pmcore.view-own-projects',
                    'pmcore.view-project-tasks',
                    'pmcore.view-project-reports',
                ];
                $role->givePermissionTo($permissions);
                $this->command->info('Updated Client role with PMCore permissions');
            }
        } catch (\Exception $e) {
            $this->command->error('Failed to update Client role: '.$e->getMessage());
        }
    }
}
