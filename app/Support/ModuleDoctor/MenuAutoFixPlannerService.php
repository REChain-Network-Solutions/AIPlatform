<?php
namespace App\Support\ModuleDoctor;
class MenuAutoFixPlannerService {
    public function plan(array $audit): array {
        return [
            'sql' => app(MenuSqlPreviewService::class)->preview('dashboard.user.module.index', 'Module'),
            'sidebar' => app(SidebarInjectorPreviewService::class)->preview('dashboard.user.module.index', 'Module'),
            'queue' => app(AutoRepairQueueService::class)->build($audit),
        ];
    }
}
