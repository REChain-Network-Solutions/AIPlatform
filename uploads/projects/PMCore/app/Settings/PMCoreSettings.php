<?php

namespace Modules\PMCore\app\Settings;

use App\Services\Settings\BaseModuleSettings;

class PMCoreSettings extends BaseModuleSettings
{
    protected string $module = 'PMCore';

    /**
     * Get module display name
     */
    public function getModuleName(): string
    {
        return __('Project Management Settings');
    }

    /**
     * Get module description
     */
    public function getModuleDescription(): string
    {
        return __('Configure project defaults, workflow rules, and management preferences');
    }

    /**
     * Get module icon
     */
    public function getModuleIcon(): string
    {
        return 'bx bx-briefcase';
    }

    /**
     * Define module settings
     */
    protected function define(): array
    {
        return [
            'project_defaults' => [
                'default_project_status' => [
                    'type' => 'select',
                    'label' => __('Default Project Status'),
                    'help' => __('Default status for new projects'),
                    'default' => 'planning',
                    'options' => [
                        'planning' => __('Planning'),
                        'in_progress' => __('In Progress'),
                        'on_hold' => __('On Hold'),
                        'completed' => __('Completed'),
                        'cancelled' => __('Cancelled'),
                    ],
                    'validation' => 'required|string',
                ],
                'default_project_priority' => [
                    'type' => 'select',
                    'label' => __('Default Project Priority'),
                    'help' => __('Default priority for new projects'),
                    'default' => 'medium',
                    'options' => [
                        'low' => __('Low'),
                        'medium' => __('Medium'),
                        'high' => __('High'),
                        'urgent' => __('Urgent'),
                    ],
                    'validation' => 'required|string',
                ],
                'default_is_billable' => [
                    'type' => 'toggle',
                    'label' => __('Projects Billable by Default'),
                    'help' => __('New projects are billable by default'),
                    'default' => true,
                    'validation' => 'boolean',
                ],
            ],

            'project_code' => [
                'auto_generate_codes' => [
                    'type' => 'toggle',
                    'label' => __('Auto-Generate Project Codes'),
                    'help' => __('Automatically generate unique codes for projects'),
                    'default' => true,
                    'validation' => 'boolean',
                ],
                'code_prefix_length' => [
                    'type' => 'number',
                    'label' => __('Code Prefix Length'),
                    'help' => __('Number of characters from project name (1-10)'),
                    'default' => '3',
                    'validation' => 'required|numeric|min:1|max:10',
                ],
                'code_separator' => [
                    'type' => 'text',
                    'label' => __('Code Separator'),
                    'help' => __('Character between prefix and number'),
                    'default' => '-',
                    'validation' => 'nullable|string|max:1',
                ],
            ],

            'timesheet_rules' => [
                'allow_future_timesheets' => [
                    'type' => 'toggle',
                    'label' => __('Allow Future Date Timesheets'),
                    'help' => __('Allow logging time for future dates'),
                    'default' => false,
                    'validation' => 'boolean',
                ],
                'max_hours_per_day' => [
                    'type' => 'number',
                    'label' => __('Maximum Hours Per Day'),
                    'help' => __('Maximum hours allowed per timesheet entry'),
                    'default' => '12',
                    'validation' => 'required|numeric|min:1|max:24',
                ],
                'require_task_assignment' => [
                    'type' => 'toggle',
                    'label' => __('Require Task Assignment'),
                    'help' => __('Timesheets must be assigned to specific tasks'),
                    'default' => false,
                    'validation' => 'boolean',
                ],
            ],

            'budget_management' => [
                'budget_warning_threshold' => [
                    'type' => 'number',
                    'label' => __('Budget Warning Threshold (%)'),
                    'help' => __('Show warning at this percentage of budget'),
                    'default' => '80',
                    'validation' => 'required|numeric|min:1|max:100',
                ],
                'budget_alert_threshold' => [
                    'type' => 'number',
                    'label' => __('Budget Alert Threshold (%)'),
                    'help' => __('Show alert at this percentage of budget'),
                    'default' => '100',
                    'validation' => 'required|numeric|min:1|max:200',
                ],
                'default_currency_symbol' => [
                    'type' => 'text',
                    'label' => __('Default Currency Symbol'),
                    'help' => __('Currency symbol for budget displays'),
                    'default' => '$',
                    'validation' => 'required|string|max:5',
                ],
            ],

            'team_management' => [
                'default_allocation_percentage' => [
                    'type' => 'number',
                    'label' => __('Default Team Member Allocation (%)'),
                    'help' => __('Default allocation percentage for new team members'),
                    'default' => '100',
                    'validation' => 'required|numeric|min:1|max:100',
                ],
                'allow_multiple_project_managers' => [
                    'type' => 'toggle',
                    'label' => __('Allow Multiple Project Managers'),
                    'help' => __('Allow projects to have multiple managers'),
                    'default' => false,
                    'validation' => 'boolean',
                ],
            ],
        ];
    }
}
