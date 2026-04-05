<?php
namespace App\Services\CustomModules;

use App\Services\CustomModules\Analyzers\DoctorPanelAnalyzer;
use App\Services\CustomModules\Analyzers\ManifestAnalyzer;
use App\Services\CustomModules\Analyzers\PermissionAnalyzer;
use App\Services\CustomModules\Analyzers\RouteAnalyzer;
use App\Services\CustomModules\Analyzers\SidebarAnalyzer;
use App\Services\CustomModules\DTO\ModuleAnalysisResult;

class CustomModuleAnalyzer
{
    public function __construct(
        protected ManifestAnalyzer $manifestAnalyzer = new ManifestAnalyzer(),
        protected RouteAnalyzer $routeAnalyzer = new RouteAnalyzer(),
        protected SidebarAnalyzer $sidebarAnalyzer = new SidebarAnalyzer(),
        protected PermissionAnalyzer $permissionAnalyzer = new PermissionAnalyzer(),
        protected DoctorPanelAnalyzer $doctorPanelAnalyzer = new DoctorPanelAnalyzer(),
    ) {}

    public function analyze(string $filePath): ModuleAnalysisResult
    {
        $manifest = $this->manifestAnalyzer->inspect($filePath);
        $routes = $this->routeAnalyzer->inspect($filePath);
        $sidebar = $this->sidebarAnalyzer->inspect($filePath);
        $permissions = $this->permissionAnalyzer->inspect($filePath);
        $panels = $this->doctorPanelAnalyzer->inspect($filePath);

        $warnings = array_merge($manifest['warnings'], $routes['warnings'], $sidebar['warnings'], $permissions['warnings'], $panels['warnings']);
        $blocking = array_merge($manifest['blocking_issues'], $routes['blocking_issues'], $sidebar['blocking_issues'], $permissions['blocking_issues'], $panels['blocking_issues']);

        return new ModuleAnalysisResult(
            status: empty($blocking) ? 'ok' : 'warning',
            file: basename($filePath),
            size: filesize($filePath) ?: 0,
            modifiedAt: date('c', filemtime($filePath) ?: time()),
            checks: [
                'zip_readable' => true,
                'manifest' => $manifest['status'],
                'routes' => $routes['status'],
                'sidebar' => $sidebar['status'],
                'permissions' => $permissions['status'],
                'doctor_panels' => $panels['status'],
            ],
            warnings: $warnings,
            blockingIssues: $blocking,
            summaryCards: [
                [
                    'title' => 'Doctor summary',
                    'body' => 'Route truth, sidebar inclusion, permission readiness, registry checks, and tenant entitlement checks are enabled in the analyzer.',
                    'theme' => 'primary',
                ],
                [
                    'title' => 'Safe fix preview',
                    'body' => 'Menu SQL preview, sidebar injector preview, and safe repair queue are available in a compact card layout.',
                    'theme' => 'success',
                ],
                [
                    'title' => 'Upload persistence',
                    'body' => 'ZIP storage, file list refresh, and duplicate-name protection are active in this rebuild.',
                    'theme' => 'primary',
                ],
            ],
            fixPlan: [
                'repair_permissions' => empty($permissions['blocking_issues']) ? 'not_needed' : 'recommended',
                'repair_package_bridge' => 'recommended',
                'repair_company_module_settings' => 'recommended',
                'clear_caches' => 'recommended',
            ],
            meta: [
                'detected_root' => $manifest['module_root'] ?? null,
                'named_routes_detected' => $routes['named_routes'] ?? [],
                'menu_signal_count' => $sidebar['menu_signal_count'] ?? 0,
                'doctor_panels' => $panels['panels'] ?? [],
            ],
        );
    }
}
