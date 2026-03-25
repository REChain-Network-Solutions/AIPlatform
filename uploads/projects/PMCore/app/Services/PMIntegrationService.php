<?php

namespace Modules\PMCore\app\Services;

use Modules\CRMCore\app\Models\Company;
use Modules\CRMCore\app\Models\Contact;
use Modules\TaskSystem\app\Models\Task;
use Nwidart\Modules\Facades\Module;

class PMIntegrationService
{
    /**
     * Check if CRMCore module is available.
     */
    public function isCRMCoreAvailable(): bool
    {
        return Module::isEnabled('CRMCore');
    }

    /**
     * Check if TaskSystem module is available.
     */
    public function isTaskSystemAvailable(): bool
    {
        return Module::isEnabled('TaskSystem');
    }

    /**
     * Check if AccountingCore module is available.
     */
    public function isAccountingCoreAvailable(): bool
    {
        return Module::isEnabled('AccountingCore');
    }

    /**
     * Check if MultiCurrency module is available.
     */
    public function isMultiCurrencyAvailable(): bool
    {
        return Module::isEnabled('MultiCurrency');
    }

    /**
     * Get available clients for project assignment.
     */
    public function getAvailableClients()
    {
        if (! $this->isCRMCoreAvailable()) {
            return collect();
        }

        return Contact::active()
            ->select('id', 'name', 'email', 'phone')
            ->orderBy('name')
            ->get();
    }

    /**
     * Get available companies for project assignment.
     */
    public function getAvailableCompanies()
    {
        if (! $this->isCRMCoreAvailable()) {
            return collect();
        }

        return Company::active()
            ->select('id', 'name', 'email', 'phone')
            ->orderBy('name')
            ->get();
    }

    /**
     * Get project managers (users with PM role).
     */
    public function getProjectManagers()
    {
        return \App\Models\User::whereHas('roles', function ($query) {
            $query->where('name', 'project_manager');
        })->orWhereHas('roles', function ($query) {
            $query->where('name', 'admin');
        })->select('id', 'name', 'email')
            ->orderBy('name')
            ->get();
    }

    /**
     * Get available team members.
     */
    public function getAvailableTeamMembers()
    {
        return \App\Models\User::active()
            ->select('id', 'name', 'email', 'designation')
            ->orderBy('name')
            ->get();
    }

    /**
     * Get project tasks count.
     */
    public function getProjectTasksCount($projectId): int
    {
        if (! $this->isTaskSystemAvailable()) {
            return 0;
        }

        return Task::where('project_id', $projectId)->count();
    }

    /**
     * Get project completed tasks count.
     */
    public function getProjectCompletedTasksCount($projectId): int
    {
        if (! $this->isTaskSystemAvailable()) {
            return 0;
        }

        return Task::where('project_id', $projectId)
            ->where('status', 'completed')
            ->count();
    }

    /**
     * Get project progress percentage.
     */
    public function getProjectProgress($projectId): float
    {
        $totalTasks = $this->getProjectTasksCount($projectId);
        if ($totalTasks === 0) {
            return 0;
        }

        $completedTasks = $this->getProjectCompletedTasksCount($projectId);

        return round(($completedTasks / $totalTasks) * 100, 2);
    }

    /**
     * Get project budget information.
     */
    public function getProjectBudgetInfo($projectId)
    {
        if (! $this->isAccountingCoreAvailable()) {
            return [
                'total_budget' => 0,
                'spent_amount' => 0,
                'remaining_budget' => 0,
                'budget_utilization' => 0,
            ];
        }

        // This would integrate with AccountingCore for actual budget tracking
        // For now, return default values
        return [
            'total_budget' => 0,
            'spent_amount' => 0,
            'remaining_budget' => 0,
            'budget_utilization' => 0,
        ];
    }

    /**
     * Get currency symbol for display.
     */
    public function getCurrencySymbol(): string
    {
        if ($this->isMultiCurrencyAvailable()) {
            // Get default currency from MultiCurrency module
            $defaultCurrency = \Modules\MultiCurrency\app\Models\Currency::where('is_default', true)->first();

            return $defaultCurrency ? $defaultCurrency->symbol : '$';
        }

        return '$';
    }

    /**
     * Format currency amount.
     */
    public function formatCurrency($amount, $currencyId = null): string
    {
        if ($this->isMultiCurrencyAvailable() && $currencyId) {
            $currency = \Modules\MultiCurrency\app\Models\Currency::find($currencyId);
            if ($currency) {
                return $currency->symbol.number_format($amount, 2);
            }
        }

        return $this->getCurrencySymbol().number_format($amount, 2);
    }

    /**
     * Check if user can manage projects.
     */
    public function canManageProjects($user = null): bool
    {
        $user = $user ?: auth()->user();

        return $user->hasRole(['admin', 'project_manager']) ||
               $user->hasPermissionTo('manage_projects');
    }

    /**
     * Check if user can view project.
     */
    public function canViewProject($projectId, $user = null): bool
    {
        $user = $user ?: auth()->user();

        if ($this->canManageProjects($user)) {
            return true;
        }

        // Check if user is a project member
        return \Modules\PMCore\app\Models\ProjectMember::where('project_id', $projectId)
            ->where('user_id', $user->id)
            ->exists();
    }

    /**
     * Get project statistics.
     */
    public function getProjectStats()
    {
        $stats = [
            'total_projects' => \Modules\PMCore\app\Models\Project::count(),
            'active_projects' => \Modules\PMCore\app\Models\Project::where('status', \Modules\PMCore\app\Enums\ProjectStatus::IN_PROGRESS->value)->count(),
            'completed_projects' => \Modules\PMCore\app\Models\Project::where('status', \Modules\PMCore\app\Enums\ProjectStatus::COMPLETED->value)->count(),
            'overdue_projects' => \Modules\PMCore\app\Models\Project::where('end_date', '<', now())
                ->whereNotIn('status', [\Modules\PMCore\app\Enums\ProjectStatus::COMPLETED->value, \Modules\PMCore\app\Enums\ProjectStatus::CANCELLED->value])
                ->count(),
        ];

        return $stats;
    }

    /**
     * Get filter data for project lists.
     */
    public function getFilterData()
    {
        return [
            'statuses' => \Modules\PMCore\app\Enums\ProjectStatus::cases(),
            'types' => \Modules\PMCore\app\Enums\ProjectType::cases(),
            'priorities' => \Modules\PMCore\app\Enums\ProjectPriority::cases(),
            'project_managers' => $this->getProjectManagers(),
            'clients' => $this->getAvailableClients(),
            'companies' => $this->getAvailableCompanies(),
        ];
    }
}
