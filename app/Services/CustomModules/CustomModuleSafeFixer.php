<?php
namespace App\Services\CustomModules;

use App\Services\CustomModules\Actions\QueueCacheClearAction;
use App\Services\CustomModules\Actions\QueueCompanyModuleSettingsRepairAction;
use App\Services\CustomModules\Actions\QueuePackageBridgeRepairAction;
use App\Services\CustomModules\Actions\QueuePermissionRepairAction;
use App\Services\CustomModules\DTO\ModuleAnalysisResult;
use App\Services\CustomModules\DTO\SafeFixPlan;

class CustomModuleSafeFixer
{
    public function __construct(
        protected QueuePermissionRepairAction $permissionAction = new QueuePermissionRepairAction(),
        protected QueuePackageBridgeRepairAction $packageBridgeAction = new QueuePackageBridgeRepairAction(),
        protected QueueCompanyModuleSettingsRepairAction $companyAction = new QueueCompanyModuleSettingsRepairAction(),
        protected QueueCacheClearAction $cacheAction = new QueueCacheClearAction(),
    ) {}

    public function plan(ModuleAnalysisResult $analysis, array $options = []): array
    {
        $selected = [
            'repair_permissions' => (bool) ($options['repair_permissions'] ?? true),
            'repair_package_bridge' => (bool) ($options['repair_package_bridge'] ?? true),
            'repair_company_module_settings' => (bool) ($options['repair_company_module_settings'] ?? true),
            'clear_caches' => (bool) ($options['clear_caches'] ?? true),
        ];

        $actions = [];
        if ($selected['repair_permissions']) $actions[] = $this->permissionAction->describe();
        if ($selected['repair_package_bridge']) $actions[] = $this->packageBridgeAction->describe();
        if ($selected['repair_company_module_settings']) $actions[] = $this->companyAction->describe();
        if ($selected['clear_caches']) $actions[] = $this->cacheAction->describe();

        $plan = new SafeFixPlan(
            selectedRepairs: $selected,
            plannedActions: $actions,
            notes: [
                'This pass prepares and exposes the safe-fix workflow with a full cumulative overlay.',
                'Recommended next pass: bind these actions to concrete SQL / service operations and log to history tables.',
            ]
        );

        return [
            'status' => 'ok',
            'message' => 'Safe fix plan generated.',
            'file' => $analysis->file,
            'selected_repairs' => $plan->selectedRepairs,
            'planned_actions' => $plan->plannedActions,
            'notes' => $plan->notes,
        ];
    }
}
