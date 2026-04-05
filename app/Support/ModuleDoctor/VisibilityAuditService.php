<?php
namespace App\Support\ModuleDoctor;

class VisibilityAuditService
{
    public function build(array $module = []): array
    {
        return $this->audit($module);
    }

    public function audit(array $module = []): array
    {
        $sidebar     = app(SidebarReadinessService::class)->inspect($module);
        $routes      = app(RouteReadinessService::class)->inspect($module);
        $permissions = app(PermissionReadinessService::class)->inspect($module);
        $package     = app(PackageBridgeReadinessService::class)->inspect($module);
        $menu        = app(MenuRegistrationAuditService::class)->inspect($module);
        $provider    = app(ProviderBootVerificationService::class)->inspect($module);

        $sections = compact('sidebar', 'routes', 'permissions', 'package', 'menu', 'provider');

        $allReady = collect($sections)->every(fn ($s) => $s['ready'] ?? false);
        $knownIssues = collect($sections)
            ->filter(fn ($s) => !($s['ready'] ?? false))
            ->map(fn ($s, $key) => ['section' => $key, 'detail' => $s['detail'] ?? 'Not ready'])
            ->values()
            ->all();

        return array_merge($sections, [
            'status'       => $allReady ? 'pass' : 'warn',
            'message'      => $allReady ? 'Module visibility checks passed.' : 'One or more visibility checks require attention.',
            'known_issues' => $knownIssues,
        ]);
    }
}
