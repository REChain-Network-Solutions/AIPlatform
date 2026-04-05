<?php

namespace App\Http\Controllers;

use App\Events\ModuleStatusChanged;
use App\Helper\Reply;
use App\Models\ModuleInstallLog;
use App\Models\ModuleSetting;
use App\Models\GlobalSetting;
use Froiden\Envato\Functions\EnvatoUpdate;
use Froiden\Envato\Traits\ModuleVerify;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\App;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Artisan;
use Illuminate\Support\Facades\File;
use Illuminate\Support\Str;
use Illuminate\Support\Facades\Schema;
use Illuminate\Support\Collection;
use App\Support\ModuleDoctor\InstallSnapshotService;
use App\Support\ModuleDoctor\RepairPlanService;
use App\Support\ModuleDoctor\VisibilityAuditService;
use App\Support\ModuleDoctor\AutoRepairQueueService;
use App\Support\ModuleDoctor\WorksuiteProfileService;
use App\Support\ModuleDoctor\SafeAutoFixExecutionService;
use App\Support\ModuleDoctor\ModuleDoctorManifest;
use App\Services\CustomModules\ModuleInstallSnapshot;
use App\Services\CustomModules\SafeFixActionExecutor;
use App\Services\CustomModules\Validators\PermissionCollisionValidator;
use App\Services\CustomModules\Validators\RouteCollisionValidator;
use App\Services\CustomModules\Validators\SecurityScanValidator;
use App\Services\CustomModules\Monitoring\InstallationEventDispatcher;
use App\Services\CustomModules\Configuration\ConfigurationManager;
use App\Services\CustomModules\Testing\AutomatedTestSuiteRunner;
use App\Services\CustomModules\Dependencies\DependencyResolver;
use App\Services\CustomModules\Marketplace\RemoteRegistryIngestor;
use App\Services\CustomModules\Marketplace\VersionResolver;
use Macellan\Zip\Zip;
use Nwidart\Modules\Facades\Module;

class CustomModuleController extends AccountBaseController
{

    use ModuleVerify;

    public function __construct()
    {
        parent::__construct();
        $this->pageTitle = 'Custom Module Installer';
        $this->activeSettingMenu = 'module_settings';
        $this->middleware(function ($request, $next) {
            abort_403(GlobalSetting::validateSuperAdmin('manage_superadmin_custom_module_settings'));

            return $next($request);
        });
    }

    /**
     * Display a listing of the resource.
     *
     * @return \Illuminate\Contracts\Foundation\Application|\Illuminate\Contracts\View\Factory|\Illuminate\Contracts\View\View
     */
    public function index()
    {
        $this->type = 'custom';
        $this->updateFilePath = config('froiden_envato.tmp_path');
        /** @phpstan-ignore-next-line */
        $this->allModules = Module::toCollection()->filter(function ($module, $key) {
            return $key !== 'UniversalBundle';
        });

        /** @phpstan-ignore-next-line */
        $this->universalBundle = Module::find('UniversalBundle');

        $this->view = 'custom-modules.ajax.custom';
        $this->doctorSummaryCards = [
            ['title' => 'Doctor summary', 'body' => 'Route truth, sidebar inclusion, permission readiness, registry checks, and tenant entitlement checks are enabled in the analyzer.', 'theme' => 'primary'],
            ['title' => 'Safe fix preview', 'body' => 'Menu SQL preview, sidebar injector preview, and safe repair queue are available without stacked full-width summary bands.', 'theme' => 'success'],
        ];
        $this->activeTab = 'custom';
        $this->plugins = collect(EnvatoUpdate::plugins());
        $this->moduleFiles = File::exists($this->updateFilePath) ? File::files($this->updateFilePath) : [];
        $this->doctorSummaryCards = $this->doctorSummaryCards ?? [];

        if (request()->ajax()) {
            $html = view($this->view, $this->data)->render();

            return Reply::dataOnly(['status' => 'success', 'html' => $html, 'title' => $this->pageTitle, 'activeTab' => $this->activeTab]);
        }

        return view('module-settings.index', $this->data);
    }

    /**
     * Show the form for creating a new resource.
     *
     * @return \Illuminate\Contracts\Foundation\Application|\Illuminate\Contracts\View\Factory|\Illuminate\Contracts\View\View
     */
    public function create()
    {
        $this->pageTitle = 'Custom Module Installer';
        $this->type = 'custom';
        $this->updateFilePath = config('froiden_envato.tmp_path');
        $this->packageChoices = $this->getPackageChoices();
        $this->moduleFiles = File::exists($this->updateFilePath) ? File::files($this->updateFilePath) : [];
        $this->doctorSummaryCards = $this->doctorSummaryCards ?? [];

        return view('custom-modules.install', $this->data);
    }

    /**
     * @param Request $request
     * @return array
     * @throws \Exception
     */
    public function store(Request $request)
    {
        $filePath = $request->filePath;
        $autoRepairMetadata = $request->boolean('auto_repair_metadata', true);
        $runtimeConfig = class_exists(ConfigurationManager::class) ? ConfigurationManager::load() : [];
        $moduleLabelForEvents = basename((string) $filePath, '.zip');

        if (class_exists(InstallationEventDispatcher::class)) {
            InstallationEventDispatcher::analysisStarted($moduleLabelForEvents, ['file_path' => $filePath]);
            InstallationEventDispatcher::securityScanStarted($moduleLabelForEvents, $filePath);
        }

        $securityValidatorEnabled = data_get($runtimeConfig, 'validators.security.enabled', true);
        $securityResult = $securityValidatorEnabled ? app(SecurityScanValidator::class)->validate($filePath) : ['valid' => true, 'issues' => [], 'checksum_sha256' => File::exists($filePath) ? hash_file('sha256', $filePath) : null, 'skipped' => true];
        if (class_exists(InstallationEventDispatcher::class)) {
            InstallationEventDispatcher::securityScanCompleted($moduleLabelForEvents, (bool) ($securityResult['valid'] ?? false), $securityResult);
        }
        if (!($securityResult['valid'] ?? false)) {
            $analysis = [
                'status' => false,
                'message' => 'Module failed security scan. Possible malicious content detected.',
                'blocking_issues' => collect($securityResult['issues'] ?? [])->pluck('message')->filter()->values()->all(),
                'warnings' => [],
                'checks' => [[
                    'label' => 'Security scan',
                    'status' => 'fail',
                    'detail' => collect($securityResult['issues'] ?? [])->pluck('code')->filter()->implode(', ') ?: 'Security scan failed',
                ]],
                'package_summary' => [
                    'uploaded_zip' => basename((string) $filePath),
                    'checksum_sha256' => $securityResult['checksum_sha256'] ?? null,
                ],
            ];
            $analysis['diagnostics_html'] = $this->renderDiagnosticsHtml($analysis);
            $this->persistInstallLog('install_blocked_security', $analysis, $filePath);

            return Reply::dataOnly([
                'status' => 'fail',
                'message' => $analysis['message'],
                'diagnostics' => $analysis['diagnostics_html'] ?? null,
                'analysis' => $analysis,
            ]);
        }

        if (class_exists(InstallationEventDispatcher::class)) {
            InstallationEventDispatcher::permissionCheckStarted($moduleLabelForEvents, $filePath);
        }
        $permissionValidatorEnabled = data_get($runtimeConfig, 'validators.permissions.enabled', true);
        $permResult = $permissionValidatorEnabled ? app(PermissionCollisionValidator::class)->validate($filePath) : ['valid' => true, 'issues' => [], 'permissions_count' => 0, 'collisions_found' => 0, 'skipped' => true];
        if (class_exists(InstallationEventDispatcher::class)) {
            InstallationEventDispatcher::permissionCheckCompleted($moduleLabelForEvents, (bool) ($permResult['valid'] ?? false), $permResult);
        }
        if (!($permResult['valid'] ?? false)) {
            $analysis = [
                'status' => false,
                'message' => 'Module has permission key collisions with existing modules.',
                'blocking_issues' => collect($permResult['issues'] ?? [])->pluck('message')->filter()->values()->all(),
                'warnings' => [],
                'checks' => [[
                    'label' => 'Permission collision check',
                    'status' => 'fail',
                    'detail' => ($permResult['collisions_found'] ?? 0) . ' collision(s) found',
                ]],
                'package_summary' => [
                    'uploaded_zip' => basename((string) $filePath),
                    'permissions_count' => $permResult['permissions_count'] ?? 0,
                ],
            ];
            $analysis['diagnostics_html'] = $this->renderDiagnosticsHtml($analysis);
            $this->persistInstallLog('install_blocked_permission_collision', $analysis, $filePath);

            return Reply::dataOnly([
                'status' => 'fail',
                'message' => $analysis['message'],
                'diagnostics' => $analysis['diagnostics_html'] ?? null,
                'analysis' => $analysis,
            ]);
        }

        if (class_exists(InstallationEventDispatcher::class)) {
            InstallationEventDispatcher::routeCheckStarted($moduleLabelForEvents, $filePath);
        }
        $routeValidatorEnabled = data_get($runtimeConfig, 'validators.routes.enabled', true);
        $routeResult = $routeValidatorEnabled ? app(RouteCollisionValidator::class)->validate($filePath) : ['valid' => true, 'issues' => [], 'routes_count' => 0, 'collisions_found' => 0, 'skipped' => true];
        if (class_exists(InstallationEventDispatcher::class)) {
            InstallationEventDispatcher::routeCheckCompleted($moduleLabelForEvents, (bool) ($routeResult['valid'] ?? false), $routeResult);
        }
        if (!($routeResult['valid'] ?? false)) {
            $analysis = [
                'status' => false,
                'message' => 'Module has route URI collisions with existing routes.',
                'blocking_issues' => collect($routeResult['issues'] ?? [])->pluck('message')->filter()->values()->all(),
                'warnings' => [],
                'checks' => [[
                    'label' => 'Route collision check',
                    'status' => 'fail',
                    'detail' => ($routeResult['collisions_found'] ?? 0) . ' collision(s) found',
                ]],
                'package_summary' => [
                    'uploaded_zip' => basename((string) $filePath),
                    'routes_checked' => $routeResult['routes_count'] ?? 0,
                ],
            ];
            $analysis['diagnostics_html'] = $this->renderDiagnosticsHtml($analysis);
            $this->persistInstallLog('install_blocked_route_collision', $analysis, $filePath);

            return Reply::dataOnly([
                'status' => 'fail',
                'message' => $analysis['message'],
                'diagnostics' => $analysis['diagnostics_html'] ?? null,
                'analysis' => $analysis,
            ]);
        }

        $analysis = $this->analyzeUploadedModule($filePath, true, $autoRepairMetadata);
        if (class_exists(DependencyResolver::class)) {
            try {
                $dependencies = app(DependencyResolver::class)->extractDependenciesFromZip($filePath);
                $analysis['package_summary']['declared_dependencies'] = $dependencies;
                $analysis['checks'][] = [
                    'label' => 'Dependency manifest',
                    'status' => empty($dependencies) ? 'pass' : 'info',
                    'detail' => empty($dependencies) ? 'No manifest dependencies declared.' : count($dependencies) . ' dependenc' . (count($dependencies) === 1 ? 'y' : 'ies') . ' declared.',
                ];
            } catch (\Throwable $e) {
                $analysis['warnings'][] = 'Dependency extraction failed: ' . $e->getMessage();
            }
        }
        $analysis = $this->applyAdvancedValidatorSignals($analysis, $filePath, $securityResult, $permResult, $routeResult);
        $analysis = $this->enrichRegistrySignals($analysis, $filePath);
        if (class_exists(InstallationEventDispatcher::class)) {
            InstallationEventDispatcher::analysisCompleted($analysis['module_name'] ?? $moduleLabelForEvents, (bool) ($analysis['status'] ?? false), $analysis);
        }

        $allowedStatuses = ['ready', 'installable-with-warnings', 'upgrade-candidate'];

        if (!($analysis['status'] ?? false) || !in_array($analysis['installability_status'] ?? 'blocked', $allowedStatuses, true)) {
            $this->persistInstallLog('install_blocked', $analysis, $filePath);

            return Reply::dataOnly([
                'status' => 'fail',
                'message' => $analysis['message'] ?? 'The uploaded package could not be installed.',
                'diagnostics' => $analysis['diagnostics_html'] ?? null,
                'analysis' => $analysis,
            ]);
        }

        File::put(public_path() . '/install-version.txt', 'complete');

        $moduleName = $analysis['module_name'];
        $moduleSourcePath = $analysis['module_path'];
        $preInstallSnapshot = app(ModuleInstallSnapshot::class)->snapshotBefore($moduleName);
        if (class_exists(InstallationEventDispatcher::class)) {
            InstallationEventDispatcher::snapshotCreated($moduleName, $preInstallSnapshot);
        }

        if ($autoRepairMetadata) {
            $repairResult = $this->persistInferredModuleConfig($moduleSourcePath, $analysis);
            if ($repairResult) {
                $analysis['auto_repairs'][] = $repairResult;
            }
        }

        $snapshotPath = null;
        if (($analysis['installability_status'] ?? '') === 'upgrade-candidate') {
            $snapshotPath = app(InstallSnapshotService::class)->snapshotModule($moduleName);
            if ($snapshotPath) {
                $analysis['auto_repairs'][] = [
                    'title' => 'Snapshot before replace',
                    'detail' => 'Backed up the installed module before replacing it.',
                    'target' => $snapshotPath,
                ];
            }
        }

        if (class_exists(InstallationEventDispatcher::class)) {
            InstallationEventDispatcher::filesMovedStarted($moduleName, ['source' => $moduleSourcePath]);
        }
        File::ensureDirectoryExists(base_path('Modules'));
        File::moveDirectory($moduleSourcePath, base_path('Modules/' . $moduleName), true);
        if (class_exists(InstallationEventDispatcher::class)) {
            InstallationEventDispatcher::filesMovedCompleted($moduleName, true, ['destination' => base_path('Modules/' . $moduleName)]);
        }

        if (class_exists(AutomatedTestSuiteRunner::class)) {
            try {
                $tester = new AutomatedTestSuiteRunner($moduleName, null);
                $preResults = $tester->runPreInstallTests($filePath);
                $analysis['pre_install_tests'] = $preResults;
            } catch (\Throwable $e) {
                $analysis['warnings'][] = 'Pre-install tests could not be completed: ' . $e->getMessage();
            }
        }

        cache()->forget('laravel-modules');
        File::deleteDirectory(storage_path('app') . '/Modules/');

        if (module_enabled($moduleName)) {
            $this->updateVersion($moduleName);
        }

        if ($moduleName == 'UniversalBundle') {
            /** @phpstan-ignore-next-line */
            $module = Module::findOrFail($moduleName);
            $module->enable();
            Artisan::call('module:migrate', array($moduleName, '--force' => true));
            event(new ModuleStatusChanged($module, 'active'));
        }

        $repairOptions = [
            'repair_permissions' => $request->boolean('repair_permissions', true),
            'repair_package_bridge' => $request->boolean('repair_package_bridge', true),
            'repair_company_module_settings' => $request->boolean('repair_company_module_settings', true),
            'clear_caches' => $request->boolean('clear_caches', true),
        ];
        if (class_exists(InstallationEventDispatcher::class)) {
            InstallationEventDispatcher::repairsStarted($moduleName, $repairOptions);
        }
        $repairExecution = app(SafeFixActionExecutor::class)->executeAll($moduleName, $repairOptions);
        if (class_exists(InstallationEventDispatcher::class)) {
            InstallationEventDispatcher::repairsCompleted($moduleName, empty($repairExecution['failed'] ?? []), $repairExecution);
        }

        $postInstall = $this->runPostInstallActions($request, $moduleName);
        if (class_exists(AutomatedTestSuiteRunner::class)) {
            try {
                $tester = new AutomatedTestSuiteRunner($moduleName, null);
                $postInstall['automated_tests'] = $tester->runPostInstallTests();
                if (($postInstall['automated_tests']['files_exist']['success'] ?? true) === false) {
                    app(ModuleInstallSnapshot::class)->restoreFrom($moduleName, $preInstallSnapshot ?? []);
                    if (class_exists(InstallationEventDispatcher::class)) {
                        InstallationEventDispatcher::installationFailed($moduleName, ['reason' => 'Automated post-install tests failed']);
                    }
                    return Reply::dataOnly([
                        'status' => 'fail',
                        'message' => 'Installation test failed after file copy. Rollback executed.',
                        'analysis' => array_merge($analysis, ['post_install' => $postInstall]),
                    ]);
                }
            } catch (\Throwable $e) {
                $postInstall['automated_tests'] = ['status' => 'warn', 'detail' => $e->getMessage()];
            }
        }
        $postInstall['safe_fix_executor'] = $repairExecution;
        if ($snapshotPath) {
            $postInstall['snapshot'] = ['status' => 'success', 'detail' => $snapshotPath];
        }
        $analysis['post_install'] = $postInstall;
        $analysis['repair_execution'] = $repairExecution;
        $analysis['diagnostics_html'] = $this->renderDiagnosticsHtml($analysis);

        $installStatus = empty($repairExecution['failed'] ?? []) ? 'installed' : 'partial_failed';
        $log = ModuleInstallLog::create([
            'module_name' => $moduleName,
            'version' => $analysis['package_summary']['declared_version'] ?? ($analysis['version'] ?? 'unknown'),
            'status' => $installStatus,
            'package_summary' => $analysis['package_summary'] ?? [],
            'checks' => $analysis['checks'] ?? [],
            'warnings' => $analysis['warnings'] ?? [],
            'blocking_issues' => $analysis['blocking_issues'] ?? [],
            'manifest_data' => $analysis,
            'validation_results' => [
                'security' => $securityResult,
                'permissions' => $permResult,
                'routes' => $routeResult,
            ],
            'validation_score' => empty($repairExecution['failed'] ?? []) ? 100 : 85,
            'pre_install_snapshot' => $preInstallSnapshot,
            'applied_repairs' => array_keys($repairExecution['executed'] ?? []),
            'repair_results' => $repairExecution,
            'module_source_hash' => File::exists($filePath) ? hash_file('sha256', $filePath) : null,
            'module_source_size' => File::exists($filePath) ? filesize($filePath) : null,
            'can_rollback' => !empty($preInstallSnapshot),
            'installed_at' => now(),
            'last_verified_at' => now(),
        ]);
        $analysis['log_id'] = $log->id;

        if (class_exists(InstallationEventDispatcher::class)) {
            InstallationEventDispatcher::installationCompleted($moduleName, ['install_id' => $log->id, 'status' => $installStatus]);
        }

        $this->persistInstallLog('install_success', $analysis, $filePath);

        $this->flushData();

        return Reply::dataOnly([
            'status' => empty($repairExecution['failed'] ?? []) ? 'success' : 'warning',
            'message' => empty($repairExecution['failed'] ?? []) ? 'Installed successfully.' : 'Installed with repair warnings.',
            'diagnostics' => $analysis['diagnostics_html'] ?? null,
            'analysis' => $analysis,
            'rollback_url' => route('custom-modules.rollback', ['install' => $log->id]),
        ]);
    }

    public function analyze(Request $request)
    {
        $analysis = $this->analyzeUploadedModule($request->filePath, false, $request->boolean('auto_repair_metadata', true));
        $analysis = $this->applyAdvancedValidatorSignals($analysis, $request->filePath);
        $analysis = $this->enrichRegistrySignals($analysis, $request->filePath);
        $log = $this->persistInstallLog('analysis', $analysis, $request->filePath);
        $analysis['log_id'] = $log?->id;

        return Reply::dataOnly([
            'status' => ($analysis['status'] ?? false) ? 'success' : 'fail',
            'message' => $analysis['message'] ?? null,
            'diagnostics' => $analysis['diagnostics_html'] ?? null,
            'analysis' => $analysis,
        ]);
    }

    public function applySafeFixes(Request $request)
    {
        $analysis = $this->analyzeUploadedModule($request->filePath, false, $request->boolean('auto_repair_metadata', true));
        $moduleName = $analysis['module_name'] ?? null;

        if (!$moduleName) {
            return Reply::dataOnly([
                'status' => 'fail',
                'message' => 'Could not determine the module name for safe autofix.',
                'diagnostics' => $analysis['diagnostics_html'] ?? null,
                'analysis' => $analysis,
            ]);
        }

        $attachMode = (string) $request->input('package_attach_mode', 'none');
        $selectedPackageIds = collect($request->input('package_ids', []))->filter()->map(fn ($id) => (int) $id)->unique()->values();

        $fixResult = app(SafeAutoFixExecutionService::class)->run($moduleName, $analysis, $attachMode, $selectedPackageIds);
        $analysis['auto_repairs'] = array_merge($analysis['auto_repairs'] ?? [], $fixResult['applied'] ?? []);
        if (!empty($fixResult['warnings'])) {
            $analysis['warnings'] = array_merge($analysis['warnings'] ?? [], $fixResult['warnings']);
        }

        $analysis['visibility_report'] = $this->auditInstalledModuleState($moduleName)['audit'] ?? ($analysis['visibility_report'] ?? []);
        $analysis['diagnostics_html'] = $this->renderDiagnosticsHtml($analysis);
        $log = $this->persistInstallLog('safe_autofix', $analysis, $request->filePath);
        $analysis['log_id'] = $log?->id;

        return Reply::dataOnly([
            'status' => 'success',
            'message' => $fixResult['detail'] ?? 'Safe autofix completed.',
            'diagnostics' => $analysis['diagnostics_html'] ?? null,
            'analysis' => $analysis,
        ]);
    }

    private function runPostInstallActions(Request $request, string $moduleName): array
    {
        $results = [
            'package_attach' => ['status' => 'skipped', 'detail' => 'No package action requested.'],
            'module_settings_seed' => ['status' => 'skipped', 'detail' => 'No company module_settings seeding requested.'],
            'permission_seed' => ['status' => 'skipped', 'detail' => 'No permission seeding requested.'],
            'runtime_enable' => ['status' => 'skipped', 'detail' => 'Module was not auto-enabled after install.'],
            'module_migrate' => ['status' => 'skipped', 'detail' => 'Module migrations were not run automatically.'],
            'post_install_audit' => ['status' => 'skipped', 'detail' => 'Post-install doctor audit was not requested.'],
            'safe_autofix' => ['status' => 'skipped', 'detail' => 'No safe autofixes were applied.'],
        ];

        $attachMode = (string) $request->input('package_attach_mode', 'none');
        $selectedPackageIds = collect($request->input('package_ids', []))->filter()->map(fn ($id) => (int) $id)->unique()->values();

        if ($attachMode !== 'none') {
            $results['package_attach'] = $this->attachModuleToPackages($moduleName, $attachMode, $selectedPackageIds);
        }

        if ($request->boolean('seed_company_module_settings')) {
            $results['module_settings_seed'] = $this->seedModuleSettingsForCompanies($moduleName, $attachMode, $selectedPackageIds);
        }

        if ($request->boolean('run_permission_seeders')) {
            $results['permission_seed'] = $this->runPermissionSeeders($moduleName);
        }

        if ($request->boolean('enable_module_after_install')) {
            $results['runtime_enable'] = $this->enableInstalledModule($moduleName);
        }

        if ($request->boolean('run_module_migrations')) {
            $results['module_migrate'] = $this->runInstalledModuleMigrations($moduleName);
        }

        if ($request->boolean('run_post_install_audit', true)) {
            $results['post_install_audit'] = $this->auditInstalledModuleState($moduleName);
        }

        if ($request->boolean('run_safe_autofix', true)) {
            $results['safe_autofix'] = app(SafeAutoFixExecutionService::class)->run($moduleName, [
                'visibility_report' => $results['post_install_audit']['audit'] ?? ($this->auditInstalledModuleState($moduleName)['audit'] ?? []),
            ], $attachMode, $selectedPackageIds);
        }

        return $results;
    }

    private function attachModuleToPackages(string $moduleName, string $attachMode, Collection $selectedPackageIds): array
    {
        if (!Schema::hasTable('packages')) {
            return ['status' => 'warn', 'detail' => 'packages table was not found, so package attachment was skipped.'];
        }

        $query = DB::table('packages')->select('id', 'module_in_package');
        if ($attachMode === 'selected') {
            if ($selectedPackageIds->isEmpty()) {
                return ['status' => 'warn', 'detail' => 'Selected-package mode was requested, but no package IDs were provided.'];
            }
            $query->whereIn('id', $selectedPackageIds->all());
        }

        $packages = $query->get();
        $updated = 0;

        foreach ($packages as $package) {
            $modules = collect(json_decode($package->module_in_package ?: '[]', true))->filter()->values();
            if (!$modules->contains($moduleName)) {
                $modules->push($moduleName);
                DB::table('packages')->where('id', $package->id)->update(['module_in_package' => $modules->values()->toJson()]);
                $updated++;
            }
        }

        return [
            'status' => 'pass',
            'detail' => $updated . ' package(s) updated' . ($attachMode === 'all' ? ' using attach-to-all mode.' : ' using selected-package mode.'),
            'updated_count' => $updated,
        ];
    }

    private function seedModuleSettingsForCompanies(string $moduleName, string $attachMode, Collection $selectedPackageIds): array
    {
        if (!Schema::hasTable('companies') || !Schema::hasTable('module_settings')) {
            return ['status' => 'warn', 'detail' => 'companies or module_settings table was not found, so company seeding was skipped.'];
        }

        $query = DB::table('companies')->select('id', 'package_id');
        if ($attachMode === 'selected' && $selectedPackageIds->isNotEmpty()) {
            $query->whereIn('package_id', $selectedPackageIds->all());
        }

        $companies = $query->get();
        $created = 0;

        foreach ($companies as $company) {
            foreach (['admin', 'employee', 'client'] as $role) {
                $exists = DB::table('module_settings')
                    ->where('company_id', $company->id)
                    ->where('module_name', $moduleName)
                    ->where('type', $role)
                    ->exists();

                if (!$exists) {
                    DB::table('module_settings')->insert([
                        'company_id' => $company->id,
                        'module_name' => $moduleName,
                        'type' => $role,
                        'status' => 'active',
                        'is_allowed' => 1,
                        'created_at' => now(),
                        'updated_at' => now(),
                    ]);
                    $created++;
                }
            }
        }

        return [
            'status' => 'pass',
            'detail' => $created . ' module_settings row(s) created across ' . $companies->count() . ' compan' . ($companies->count() === 1 ? 'y.' : 'ies.'),
            'created_count' => $created,
        ];
    }

    private function runPermissionSeeders(string $moduleName): array
    {
        try {
            Artisan::call('module:seed', [$moduleName, '--force' => true]);
            $output = trim(Artisan::output());

            return [
                'status' => 'pass',
                'detail' => $output ?: 'module:seed executed successfully.',
            ];
        }
        catch (\Throwable $e) {
            return [
                'status' => 'warn',
                'detail' => 'module:seed could not be executed automatically: ' . $e->getMessage(),
            ];
        }
    }


    private function enableInstalledModule(string $moduleName): array
    {
        try {
            $module = Module::find($moduleName);
            if (!$module) {
                return ['status' => 'warn', 'detail' => 'Module could not be discovered by laravel-modules after file install.'];
            }

            $module->enable();
            event(new ModuleStatusChanged($module, 'active'));

            return ['status' => 'pass', 'detail' => 'Module was enabled successfully.'];
        }
        catch (\Throwable $e) {
            return ['status' => 'warn', 'detail' => 'Module enable step failed: ' . $e->getMessage()];
        }
    }

    private function runInstalledModuleMigrations(string $moduleName): array
    {
        try {
            Artisan::call('module:migrate', [$moduleName, '--force' => true]);
            $output = trim(Artisan::output());

            return [
                'status' => 'pass',
                'detail' => $output ?: 'module:migrate executed successfully.',
            ];
        }
        catch (\Throwable $e) {
            return [
                'status' => 'warn',
                'detail' => 'module:migrate could not be executed automatically: ' . $e->getMessage(),
            ];
        }
    }

    private function auditInstalledModuleState(string $moduleName): array
    {
        $modulePath = base_path('Modules/' . $moduleName);
        $module = Module::find($moduleName);
        $routeFileCount = 0;
        foreach (['Routes/web.php', 'Routes/api.php'] as $routeFile) {
            if (File::exists($modulePath . '/' . $routeFile)) {
                $routeFileCount++;
            }
        }

        $audit = [
            'module_path_exists' => File::isDirectory($modulePath),
            'module_discovered' => (bool) $module,
            'module_enabled' => $module ? $module->isEnabled() : false,
            'route_files' => $routeFileCount,
            'migration_files' => count(File::glob($modulePath . '/Database/Migrations/*.php') ?: []),
            'package_links' => Schema::hasTable('packages')
                ? DB::table('packages')->where('module_in_package', 'like', '%"' . $moduleName . '"%')->count()
                : null,
            'module_settings_rows' => Schema::hasTable('module_settings')
                ? DB::table('module_settings')->where('module_name', $moduleName)->count()
                : null,
            'permission_seeders' => count(File::glob($modulePath . '/Database/Seeders/*Permission*.php') ?: []) + count(File::glob($modulePath . '/Database/Seeders/*Role*.php') ?: []),
        ];

        $failed = [];
        if (!$audit['module_path_exists']) {
            $failed[] = 'module_path';
        }
        if (!$audit['module_discovered']) {
            $failed[] = 'discovery';
        }
        if (!$audit['module_enabled']) {
            $failed[] = 'enabled';
        }

        $detail = 'Path: ' . ($audit['module_path_exists'] ? 'yes' : 'no')
            . '; discovered: ' . ($audit['module_discovered'] ? 'yes' : 'no')
            . '; enabled: ' . ($audit['module_enabled'] ? 'yes' : 'no')
            . '; routes: ' . $audit['route_files']
            . '; migrations: ' . $audit['migration_files']
            . '; package links: ' . ($audit['package_links'] ?? 'n/a')
            . '; module_settings: ' . ($audit['module_settings_rows'] ?? 'n/a')
            . '; permission seeders: ' . $audit['permission_seeders'] . '.';

        return [
            'status' => count($failed) ? 'warn' : 'pass',
            'detail' => $detail,
            'audit' => $audit,
            'failed_checks' => $failed,
        ];
    }

    private function analyzeUploadedModule($filePath, $prepareForInstall = false, $autoRepairMetadata = true)
    {
        $analysis = [
            'status' => false,
            'message' => null,
            'module_name' => null,
            'module_path' => null,
            'detected_type' => 'unknown',
            'installability_status' => 'blocked',
            'recommended_mode' => 'repair',
            'package_summary' => [],
            'blocking_issues' => [],
            'warnings' => [],
            'checks' => [],
            'suggested_action' => 'Review the blocking issues below and upload a corrected package.',
        ];

        if (!extension_loaded('zip')) {
            $analysis['blocking_issues'][] = '<b>PHP-ZIP</b> extension is missing on your server. Please install the extension.';
            $analysis['message'] = 'Module analyzer could not start because PHP-ZIP is missing.';
            $analysis['diagnostics_html'] = $this->renderDiagnosticsHtml($analysis);

            return $analysis;
        }

        if (blank($filePath) || !File::exists($filePath)) {
            $analysis['blocking_issues'][] = 'Uploaded file path was empty or the ZIP no longer exists on disk.';
            $analysis['message'] = 'The uploaded ZIP could not be found.';
            $analysis['diagnostics_html'] = $this->renderDiagnosticsHtml($analysis);

            return $analysis;
        }

        File::deleteDirectory(storage_path('app/Modules'));
        File::deleteDirectory(storage_path('app/ModuleAnalyzer'));
        File::ensureDirectoryExists(storage_path('app/Modules'));
        File::ensureDirectoryExists(storage_path('app/ModuleAnalyzer'));

        $zip = Zip::open($filePath);
        $zipName = $this->getZipName($filePath);

        if (str_contains($zipName, 'codecanyon-')) {
            $zipName = $this->unzipCodecanyon($zip);
        }
        else {
            $zip->extract(storage_path('app/Modules'));
        }

        $moduleName = Str::replaceLast('.zip', '', $zipName);
        $modulePath = $this->resolveExtractedModulePath($moduleName);
        $versionFile = base_path('version.txt');
        $currentVersion = File::exists($versionFile) ? trim(File::get($versionFile)) : null;
        $envatoItemId = (string) config('froiden_envato.envato_item_id');
        $productName = (string) config('froiden_envato.envato_product_name');
        $appName = str_replace('-new', '', $productName);
        $topLevelDirectories = File::directories(storage_path('app/Modules'));
        $topLevelFiles = File::files(storage_path('app/Modules'));

        $analysis['module_name'] = $moduleName;
        $analysis['module_path'] = $modulePath;
        $analysis['package_summary']['uploaded_zip'] = basename($filePath);
        $analysis['package_summary']['current_worksuite_version'] = $currentVersion ?: 'unknown';
        $analysis['checks'][] = ['label' => 'ZIP readable', 'status' => 'pass', 'detail' => basename($filePath)];
        $analysis['checks'][] = ['label' => 'Archive root shape', 'status' => count($topLevelDirectories) <= 1 ? 'pass' : 'warn', 'detail' => count($topLevelDirectories) . ' top-level director' . (count($topLevelDirectories) === 1 ? 'y' : 'ies') . ', ' . count($topLevelFiles) . ' top-level file(s)'];

        if (!$modulePath || !File::exists($modulePath)) {
            $analysis['detected_type'] = $this->detectOverlayType(storage_path('app/Modules'));
            $analysis['blocking_issues'][] = 'Could not find an installable module root in storage/app/Modules.';
            $analysis['checks'][] = ['label' => 'Extracted module folder', 'status' => 'fail', 'detail' => 'No top-level module directory with Config/config.php or module.json could be resolved.'];
            $analysis['message'] = $analysis['detected_type'] === 'overlay-delta'
                ? 'The ZIP looks like an overlay/delta package, not a native installable module.'
                : 'The ZIP extracted, but its module root could not be detected.';
            $analysis['recommended_mode'] = $analysis['detected_type'] === 'overlay-delta' ? 'overlay' : 'repair';
            $analysis['installability_status'] = $analysis['detected_type'] === 'overlay-delta' ? 'overlay-only' : 'blocked';
            $analysis['suggested_action'] = $analysis['detected_type'] === 'overlay-delta'
                ? 'Import this ZIP as an overlay or donor pack instead of installing it through the native module installer.'
                : 'Rebuild the ZIP so the module folder is at the top level, or import it as an overlay package instead.';
            $analysis['diagnostics_html'] = $this->renderDiagnosticsHtml($analysis);

            if (!$prepareForInstall) {
                File::deleteDirectory(storage_path('app/Modules'));
            }

            return $analysis;
        }

        $analysis['checks'][] = ['label' => 'Extracted module folder', 'status' => 'pass', 'detail' => basename($modulePath)];

        $moduleJsonPath = $modulePath . '/module.json';
        $configPath = $modulePath . '/Config/config.php';
        $providerFiles = File::glob($modulePath . '/Providers/*ServiceProvider.php') ?: [];
        $routeFiles = array_values(array_filter([
            $modulePath . '/Routes/web.php',
            $modulePath . '/Routes/api.php',
        ], fn ($path) => File::exists($path)));
        $migrationFiles = File::glob($modulePath . '/Database/Migrations/*.php') ?: [];
        $langFiles = File::glob($modulePath . '/Resources/lang/*') ?: [];
        $hasViews = File::isDirectory($modulePath . '/Resources/views');
        $moduleJson = File::exists($moduleJsonPath) ? json_decode(File::get($moduleJsonPath), true) : [];
        $moduleJson = is_array($moduleJson) ? $moduleJson : [];

        if (File::exists($moduleJsonPath) && File::exists($configPath)) {
            $analysis['detected_type'] = 'worksuite-module';
        }
        elseif (File::exists($moduleJsonPath)) {
            $analysis['detected_type'] = 'nwidart-module';
        }
        else {
            $analysis['detected_type'] = 'legacy-worksuite-module';
        }

        $analysis['package_summary']['detected_type'] = $analysis['detected_type'];
        $analysis['package_summary']['module_root'] = basename($modulePath);
        $analysis['package_summary']['declared_version'] = $moduleJson['version'] ?? null;
        $analysis['package_summary']['declared_alias'] = $moduleJson['alias'] ?? null;

        if (File::exists($moduleJsonPath)) {
            $analysis['checks'][] = ['label' => 'module.json manifest', 'status' => 'pass', 'detail' => 'Found'];
        }
        else {
            $analysis['checks'][] = ['label' => 'module.json manifest', 'status' => 'warn', 'detail' => 'Missing'];
            $analysis['warnings'][] = 'module.json is missing. The package may still install if Config/config.php is valid, but it is not a clean nWidart module.';
        }

        if (File::exists($configPath)) {
            $analysis['checks'][] = ['label' => 'Config/config.php', 'status' => 'pass', 'detail' => 'Found'];
            $config = include $configPath;
            $config = is_array($config) ? $config : [];
        }
        else {
            $analysis['checks'][] = ['label' => 'Config/config.php', 'status' => 'fail', 'detail' => 'Missing'];
            $analysis['blocking_issues'][] = 'Config/config.php is missing from the extracted module.';
            $config = [];
        }

        $analysis['checks'][] = ['label' => 'Service providers', 'status' => count($providerFiles) ? 'pass' : 'warn', 'detail' => count($providerFiles) ? count($providerFiles) . ' provider file(s) found' : 'No *ServiceProvider.php file found'];
        if (!count($providerFiles)) {
            $analysis['warnings'][] = 'No module ServiceProvider file was detected in Providers/. Runtime boot may fail even if install succeeds.';
        }

        $analysis['checks'][] = ['label' => 'Routes', 'status' => count($routeFiles) ? 'pass' : 'warn', 'detail' => count($routeFiles) ? implode(', ', array_map('basename', $routeFiles)) : 'No Routes/web.php or Routes/api.php found'];
        if (!count($routeFiles)) {
            $analysis['warnings'][] = 'No route file was detected. The module may install but remain invisible in the UI.';
        }

        $analysis['checks'][] = ['label' => 'Migrations', 'status' => count($migrationFiles) ? 'pass' : 'warn', 'detail' => count($migrationFiles) ? count($migrationFiles) . ' migration file(s) found' : 'No migration files found'];
        if (!count($migrationFiles)) {
            $analysis['warnings'][] = 'No migrations were found. That is fine for UI-only modules, but data features may not work.';
        }

        $analysis['checks'][] = ['label' => 'Views or language resources', 'status' => ($hasViews || count($langFiles)) ? 'pass' : 'warn', 'detail' => $hasViews ? 'Views folder found' : (count($langFiles) ? 'Language files found' : 'No obvious UI resources found')];

        $menuSignals = $this->inspectMenuSignals($modulePath, $routeFiles);
        $analysis['package_summary']['menu_signal_count'] = $menuSignals['count'];
        $analysis['checks'][] = [
            'label' => 'Menu readiness',
            'status' => $menuSignals['status'],
            'detail' => $menuSignals['detail'],
        ];
        if (!empty($menuSignals['warning'])) {
            $analysis['warnings'][] = $menuSignals['warning'];
        }

        $permissionSignals = $this->inspectPermissionSignals($modulePath);
        $analysis['package_summary']['permission_signal_count'] = $permissionSignals['count'];
        $analysis['checks'][] = [
            'label' => 'Permission readiness',
            'status' => $permissionSignals['status'],
            'detail' => $permissionSignals['detail'],
        ];
        if (!empty($permissionSignals['warning'])) {
            $analysis['warnings'][] = $permissionSignals['warning'];
        }

        $packageBridgeSignals = $this->inspectPackageBridgeSignals($modulePath);
        $analysis['package_summary']['package_bridge_count'] = $packageBridgeSignals['count'];
        $analysis['checks'][] = [
            'label' => 'Package bridge readiness',
            'status' => $packageBridgeSignals['status'],
            'detail' => $packageBridgeSignals['detail'],
        ];
        if (!empty($packageBridgeSignals['warning'])) {
            $analysis['warnings'][] = $packageBridgeSignals['warning'];
        }

        $visibilityReport = app(VisibilityAuditService::class)->build([
            'module_name' => $moduleName,
            'module_path' => $modulePath,
            'sidebar' => $menuSignals,
            'routes' => ['route_names' => $menuSignals['routes'] ?? [], 'status' => count($routeFiles) ? 'pass' : 'warn', 'detail' => count($routeFiles) ? 'Routes detected' : 'No routes detected'],
            'permissions' => ['permission_keys' => $permissionSignals['keys'] ?? [], 'seeders' => $permissionSignals['seeders'] ?? [], 'status' => $permissionSignals['status'], 'detail' => $permissionSignals['detail']],
            'package_bridge' => ['status' => $packageBridgeSignals['status'], 'detail' => $packageBridgeSignals['detail']],
        ]);
        $analysis['visibility_report'] = $visibilityReport;
        $analysis['repair_queue'] = app(AutoRepairQueueService::class)->queue($visibilityReport);
        $analysis['doctor_manifest'] = app(ModuleDoctorManifest::class)->build();
        $analysis['doctor_report'] = app(\App\Support\ModuleDoctor\ModuleDoctorReportAssembler::class)->build($analysis);
        $analysis['checks'][] = [
            'label' => 'User menu visibility',
            'status' => ($visibilityReport['final']['status'] ?? 'warn') === 'fail' ? 'fail' : (($visibilityReport['final']['status'] ?? 'warn') === 'pass' ? 'pass' : 'warn'),
            'detail' => $visibilityReport['final']['detail'] ?? 'Visibility not evaluated.',
        ];
        foreach (($visibilityReport['known_issues'] ?? []) as $issue) {
            $analysis['warnings'][] = $issue['title'] . ': ' . $issue['detail'];
        }

        if (empty($config)) {
            $analysis['message'] = 'This package is missing the required module configuration file.';
            $analysis['recommended_mode'] = 'repair';
            $analysis['installability_status'] = 'blocked';
            $analysis['suggested_action'] = 'Add Config/config.php, or stage this ZIP as an overlay package instead of installing it directly.';
            $analysis['diagnostics_html'] = $this->renderDiagnosticsHtml($analysis);

            if (!$prepareForInstall) {
                File::deleteDirectory(storage_path('app/Modules'));
            }

            return $analysis;
        }

        $inferredMetadata = $this->inferWorksuiteModuleMetadata($config, $currentVersion, $envatoItemId, $productName);
        if (!empty($inferredMetadata)) {
            $analysis['package_summary']['inferred_parent_metadata'] = implode(', ', array_keys($inferredMetadata));
        }

        $parentEnvatoId = (string) ($config['parent_envato_id'] ?? ($inferredMetadata['parent_envato_id'] ?? ''));
        $parentProductName = (string) ($config['parent_product_name'] ?? ($inferredMetadata['parent_product_name'] ?? ''));
        $parentMinVersion = (string) ($config['parent_min_version'] ?? ($inferredMetadata['parent_min_version'] ?? ''));

        if (blank((string) ($config['parent_envato_id'] ?? '')) && !blank($parentEnvatoId) && $autoRepairMetadata) {
            $analysis['warnings'][] = 'Config/config.php is missing parent_envato_id. The installer can infer and write the current Worksuite product ID during install.';
            $analysis['checks'][] = ['label' => 'Parent product ID', 'status' => 'warn', 'detail' => 'Missing in config; inferred ' . $parentEnvatoId];
        }
        elseif (blank($parentEnvatoId)) {
            $analysis['blocking_issues'][] = 'Config/config.php does not define parent_envato_id.';
            $analysis['checks'][] = ['label' => 'Parent product ID', 'status' => 'fail', 'detail' => 'parent_envato_id missing'];
        }
        elseif ($parentEnvatoId !== $envatoItemId) {
            $analysis['blocking_issues'][] = 'This module targets Envato product ID <b>' . e($parentEnvatoId) . '</b>, but this installation is <b>' . e($envatoItemId) . '</b>.';
            $analysis['checks'][] = ['label' => 'Parent product ID', 'status' => 'fail', 'detail' => 'Expected ' . $envatoItemId . ', found ' . $parentEnvatoId];
        }
        else {
            $analysis['checks'][] = ['label' => 'Parent product ID', 'status' => empty($config['parent_envato_id']) ? 'warn' : 'pass', 'detail' => $parentEnvatoId];
        }

        if (blank((string) ($config['parent_product_name'] ?? '')) && !blank($parentProductName) && $autoRepairMetadata) {
            $analysis['warnings'][] = 'Config/config.php is missing parent_product_name. The installer can infer and write the current Worksuite product name during install.';
            $analysis['checks'][] = ['label' => 'Parent product name', 'status' => 'warn', 'detail' => 'Missing in config; inferred ' . $parentProductName];
        }
        elseif (blank($parentProductName)) {
            $analysis['blocking_issues'][] = 'Config/config.php does not define parent_product_name.';
            $analysis['checks'][] = ['label' => 'Parent product name', 'status' => 'fail', 'detail' => 'parent_product_name missing'];
        }
        elseif ($parentProductName !== $productName) {
            $analysis['blocking_issues'][] = 'This package was built for <b>' . e($parentProductName) . '</b>, but the current product is <b>' . e($productName) . '</b>.';
            $analysis['checks'][] = ['label' => 'Parent product name', 'status' => 'fail', 'detail' => 'Expected ' . $productName . ', found ' . $parentProductName];
        }
        else {
            $analysis['checks'][] = ['label' => 'Parent product name', 'status' => empty($config['parent_product_name']) ? 'warn' : 'pass', 'detail' => $parentProductName];
        }

        if (blank((string) ($config['parent_min_version'] ?? '')) && !blank($parentMinVersion) && $autoRepairMetadata) {
            $analysis['warnings'][] = 'Config/config.php is missing parent_min_version. The installer can infer and write a Worksuite base version during install.';
            $analysis['checks'][] = ['label' => 'Minimum Worksuite version', 'status' => 'warn', 'detail' => 'Missing in config; inferred ' . $parentMinVersion];
        }
        elseif (blank($parentMinVersion)) {
            $analysis['blocking_issues'][] = 'Config/config.php does not define parent_min_version.';
            $analysis['checks'][] = ['label' => 'Minimum Worksuite version', 'status' => 'fail', 'detail' => 'parent_min_version missing'];
        }
        elseif ($currentVersion && version_compare($currentVersion, $parentMinVersion, '<')) {
            $analysis['blocking_issues'][] = 'This module requires <b>' . e($appName) . '</b> version <b>' . e($parentMinVersion) . '</b> or later, but the current installation is <b>' . e($currentVersion) . '</b>.';
            $analysis['checks'][] = ['label' => 'Minimum Worksuite version', 'status' => 'fail', 'detail' => 'Current ' . $currentVersion . ', requires ' . $parentMinVersion];
        }
        else {
            $analysis['checks'][] = ['label' => 'Minimum Worksuite version', 'status' => empty($config['parent_min_version']) ? 'warn' : 'pass', 'detail' => $parentMinVersion ?: 'No version.txt found'];
        }

        $installedModule = Module::find($moduleName);
        if ($installedModule) {
            $analysis['warnings'][] = 'A module with this name is already registered. This upload looks like an upgrade candidate rather than a first-time install.';
            $analysis['checks'][] = ['label' => 'Existing module conflict', 'status' => 'warn', 'detail' => $moduleName . ' is already present in the module registry'];
            $analysis['package_summary']['installed_module'] = $moduleName;
            $analysis['package_summary']['installed_version'] = $this->getInstalledModuleVersion($moduleName);
        }
        else {
            $analysis['checks'][] = ['label' => 'Existing module conflict', 'status' => 'pass', 'detail' => 'No installed module with this name detected'];
        }

        if (count($migrationFiles)) {
            $existingMigrationNames = collect(File::glob(database_path('migrations/*.php')) ?: [])->map(fn ($path) => basename($path))->all();
            $conflicts = collect($migrationFiles)->map(fn ($path) => basename($path))->filter(fn ($name) => in_array($name, $existingMigrationNames))->values()->all();
            if (count($conflicts)) {
                $analysis['warnings'][] = 'Migration filename conflict detected: ' . e(implode(', ', $conflicts));
                $analysis['checks'][] = ['label' => 'Migration filename conflicts', 'status' => 'warn', 'detail' => implode(', ', $conflicts)];
            }
            else {
                $analysis['checks'][] = ['label' => 'Migration filename conflicts', 'status' => 'pass', 'detail' => 'No conflicts detected'];
            }
        }

        $analysis = $this->finalizeInstallability($analysis);
        $planner = app(RepairPlanService::class);
        $analysis['repair_plan'] = $planner->build($analysis);
        $analysis['risk_profile'] = $planner->riskProfile($analysis);
        $analysis['diagnostics_html'] = $this->renderDiagnosticsHtml($analysis);

        if (!$prepareForInstall) {
            File::deleteDirectory(storage_path('app/Modules'));
        }

        return $analysis;
    }

    private function resolveExtractedModulePath($moduleName)
    {
        $directPath = storage_path('app/Modules/' . $moduleName);
        if (File::exists($directPath)) {
            return $directPath;
        }

        $topLevelDirectories = File::directories(storage_path('app/Modules'));
        foreach ($topLevelDirectories as $directory) {
            if (File::exists($directory . '/Config/config.php') || File::exists($directory . '/module.json')) {
                return $directory;
            }

            foreach (File::directories($directory) as $childDirectory) {
                if (File::exists($childDirectory . '/Config/config.php') || File::exists($childDirectory . '/module.json')) {
                    return $childDirectory;
                }
            }
        }

        return null;
    }

    private function detectOverlayType($rootPath)
    {
        $commonOverlayPaths = [
            $rootPath . '/app',
            $rootPath . '/resources',
            $rootPath . '/routes',
            $rootPath . '/database',
            $rootPath . '/config',
        ];

        foreach ($commonOverlayPaths as $candidate) {
            if (File::exists($candidate)) {
                return 'overlay-delta';
            }
        }

        foreach (File::directories($rootPath) as $directory) {
            foreach (['app', 'resources', 'routes', 'database', 'config'] as $nested) {
                if (File::exists($directory . '/' . $nested)) {
                    return 'overlay-delta';
                }
            }
        }

        return 'unknown';
    }

    private function inspectMenuSignals(string $modulePath, array $routeFiles): array
    {
        $candidates = [];

        foreach ([
            $modulePath . '/Resources/views/sections/sidebar.blade.php',
            $modulePath . '/Resources/views/sections/menu.blade.php',
            $modulePath . '/Config/menu.php',
            $modulePath . '/menu.php',
        ] as $candidate) {
            if (File::exists($candidate)) {
                $candidates[] = basename($candidate);
            }
        }

        $namedRouteHits = 0;
        foreach ($routeFiles as $routeFile) {
            $contents = File::get($routeFile);
            $namedRouteHits += substr_count($contents, '->name(') + substr_count($contents, "name('") + substr_count($contents, 'name("');
        }

        $count = count($candidates) + $namedRouteHits;

        if ($count > 0) {
            return [
                'status' => 'pass',
                'detail' => trim(($candidates ? 'Menu/view files: ' . implode(', ', $candidates) . '. ' : '') . ($namedRouteHits ? $namedRouteHits . ' named route hint(s) detected.' : '')),
                'warning' => null,
                'count' => $count,
            ];
        }

        return [
            'status' => 'warn',
            'detail' => 'No obvious menu file or named route hint was detected.',
            'warning' => 'No obvious menu registration signal was found. The module may install correctly but still not appear in the sidebar until menu wiring is added.',
            'count' => 0,
        ];
    }

    private function inspectPermissionSignals(string $modulePath): array
    {
        $permissionFiles = [];
        foreach (File::glob($modulePath . '/Database/Seeders/*.php') ?: [] as $seeder) {
            $name = basename($seeder);
            $contents = File::get($seeder);
            if (Str::contains($name, ['Permission', 'Role']) || Str::contains($contents, ['permission', 'role_permission', 'manage_', 'add_', 'edit_', 'delete_'])) {
                $permissionFiles[] = $name;
            }
        }

        $configPath = $modulePath . '/Config/config.php';
        if (File::exists($configPath)) {
            $contents = File::get($configPath);
            if (Str::contains($contents, ['permissions', 'roles'])) {
                $permissionFiles[] = 'config.php permission hints';
            }
        }

        $permissionFiles = array_values(array_unique($permissionFiles));

        if (count($permissionFiles)) {
            return [
                'status' => 'pass',
                'detail' => implode(', ', $permissionFiles),
                'warning' => null,
                'count' => count($permissionFiles),
            ];
        }

        return [
            'status' => 'warn',
            'detail' => 'No permission seeder or permission hint was detected.',
            'warning' => 'No permission seeder or permission config was detected. Install may succeed, but roles could still be unable to access the module until permissions are added manually.',
            'count' => 0,
        ];
    }

    private function inspectPackageBridgeSignals(string $modulePath): array
    {
        $signalFiles = [];
        foreach (File::glob($modulePath . '/**/*.php') ?: [] as $phpFile) {
            // glob ** may not recurse everywhere on all hosts; fall through to iterator below if empty.
            $signalFiles[] = $phpFile;
        }
        if (!count($signalFiles)) {
            $signalFiles = [];
            foreach (File::allFiles($modulePath) as $file) {
                if ($file->getExtension() === 'php') {
                    $signalFiles[] = $file->getPathname();
                }
            }
        }

        $hits = [];
        foreach ($signalFiles as $phpFile) {
            $contents = File::get($phpFile);
            if (Str::contains($contents, ['module_settings', 'CustomModulePermission', 'PackageObserver', 'module_in_package'])) {
                $hits[] = basename($phpFile);
            }
        }

        $hits = array_values(array_unique($hits));

        if (count($hits)) {
            return [
                'status' => 'pass',
                'detail' => 'Bridge hints found in: ' . implode(', ', array_slice($hits, 0, 5)),
                'warning' => null,
                'count' => count($hits),
            ];
        }

        return [
            'status' => 'warn',
            'detail' => 'No package/module bridge hint was detected in the module files.',
            'warning' => 'No package bridge hint was found. The module may install on disk but still require manual package assignment and module_settings seeding before customers can use it.',
            'count' => 0,
        ];
    }

    private function getInstalledModuleVersion($moduleName)
    {
        $installedModuleJson = base_path('Modules/' . $moduleName . '/module.json');
        if (File::exists($installedModuleJson)) {
            $payload = json_decode(File::get($installedModuleJson), true);
            if (is_array($payload) && !empty($payload['version'])) {
                return $payload['version'];
            }
        }

        $installedConfig = base_path('Modules/' . $moduleName . '/Config/config.php');
        if (File::exists($installedConfig)) {
            $payload = include $installedConfig;
            if (is_array($payload) && !empty($payload['version'])) {
                return $payload['version'];
            }
        }

        return 'unknown';
    }

    private function finalizeInstallability(array $analysis)
    {
        $analysis['status'] = count($analysis['blocking_issues']) === 0;

        if (($analysis['detected_type'] ?? null) === 'overlay-delta') {
            $analysis['status'] = false;
            $analysis['installability_status'] = 'overlay-only';
            $analysis['recommended_mode'] = 'overlay';
            $analysis['message'] = 'This ZIP is an overlay/delta pack. It should be staged for import, not installed as a native module.';
            $analysis['suggested_action'] = 'Use an overlay importer or move these files into a review/donor area before merging into the base app.';

            return $analysis;
        }

        if (!$analysis['status']) {
            $analysis['installability_status'] = 'blocked';
            $analysis['recommended_mode'] = $this->suggestModeForAnalysis($analysis);
            $analysis['message'] = 'Package analysis found blocking issues. Review the report below.';
            $analysis['suggested_action'] = $this->suggestActionForAnalysis($analysis);

            return $analysis;
        }

        if (($analysis['package_summary']['installed_module'] ?? null)) {
            $analysis['installability_status'] = 'upgrade-candidate';
            $analysis['recommended_mode'] = 'upgrade';
            $analysis['message'] = 'Package analysis passed. This ZIP looks like an upgrade candidate for an existing module.';
            $analysis['suggested_action'] = 'Review warnings, back up the current module, then continue as an upgrade instead of a blind fresh install.';

            return $analysis;
        }

        if (!empty($analysis['warnings'])) {
            $analysis['installability_status'] = 'installable-with-warnings';
            $analysis['recommended_mode'] = 'install';
            $analysis['message'] = 'Package analysis passed with warnings. The module can install, but follow-up work may be needed.';
            $analysis['suggested_action'] = 'Install now if the warnings are acceptable, then review menu, package, and migration follow-up.';

            return $analysis;
        }

        $analysis['installability_status'] = 'ready';
        $analysis['recommended_mode'] = 'install';
        $analysis['message'] = 'Package analysis passed. The module is ready to install.';
        $analysis['suggested_action'] = 'Install now.';

        return $analysis;
    }

    private function suggestModeForAnalysis(array $analysis)
    {
        $issues = implode(' ', $analysis['blocking_issues'] ?? []);

        if (Str::contains($issues, ['parent_envato_id', 'product ID', 'product is'])) {
            return 'overlay';
        }

        if (Str::contains($issues, ['parent_min_version', 'requires'])) {
            return 'repair';
        }

        if (Str::contains($issues, ['Config/config.php', 'module root'])) {
            return 'repair';
        }

        return 'repair';
    }

    private function suggestActionForAnalysis(array $analysis)
    {
        $issues = implode(' ', $analysis['blocking_issues'] ?? []);

        if (Str::contains($issues, ['parent_envato_id', 'product ID', 'product is'])) {
            return 'This ZIP belongs to another product line. Use the correct Worksuite module build or treat it as an overlay/donor package for manual conversion.';
        }

        if (Str::contains($issues, ['parent_min_version', 'requires'])) {
            return 'Upgrade Worksuite first, or rebuild the module against the current Worksuite version before retrying.';
        }

        if (Str::contains($issues, ['Config/config.php', 'module root'])) {
            return 'Rebuild the ZIP with a valid module root and Config/config.php, or stage it as an overlay package instead.';
        }

        return 'Fix the blocking checks shown above, then re-run Analyze ZIP before installing.';
    }


private function getPackageChoices(): Collection
{
    if (!Schema::hasTable('packages')) {
        return collect();
    }

    $nameColumn = $this->detectPackageNameColumn();
    if (!$nameColumn) {
        return DB::table('packages')->select('id')->orderBy('id')->get()->map(function ($package) {
            $package->package_label = 'Package #' . $package->id;
            return $package;
        });
    }

    return DB::table('packages')
        ->select('id', $nameColumn . ' as package_label')
        ->orderBy('id')
        ->get()
        ->map(function ($package) {
            $package->package_label = $package->package_label ?: ('Package #' . $package->id);
            return $package;
        });
}

private function detectPackageNameColumn(): ?string
{
    foreach (['package', 'package_name', 'name', 'title'] as $column) {
        if (Schema::hasColumn('packages', $column)) {
            return $column;
        }
    }

    return null;
}

    private function renderDiagnosticsHtml(array $analysis)
    {
        $statusMap = [
            'ready' => ['success', 'Ready to install'],
            'installable-with-warnings' => ['warning', 'Installable with warnings'],
            'upgrade-candidate' => ['info', 'Upgrade candidate'],
            'overlay-only' => ['secondary', 'Overlay only'],
            'blocked' => ['danger', 'Not installable'],
        ];

        [$statusBadgeClass, $statusText] = $statusMap[$analysis['installability_status'] ?? 'blocked'] ?? ['danger', 'Not installable'];
        $recommendedMode = $analysis['recommended_mode'] ?? 'repair';
        $checkSummary = $this->renderCheckSummary($analysis['checks'] ?? []);

        $html = '<div class="module-diagnostic-report border rounded p-3 mt-3">';
        $html .= '<div class="d-flex flex-wrap justify-content-between align-items-start mb-3">';
        $html .= '<div><h5 class="mb-1">ZIP Analysis</h5><p class="mb-0 text-muted">Package: <strong>' . e($analysis['module_name'] ?? 'Unknown') . '</strong></p></div>';
        $html .= '<span class="badge badge-' . $statusBadgeClass . ' p-2">' . e($statusText) . '</span>';
        $html .= '</div>';
        $html .= '<div class="row mb-3">';
        $html .= '<div class="col-md-4 mb-2"><div class="border rounded p-2 h-100"><div class="text-muted f-12 text-uppercase">Detected type</div><div><strong>' . e($analysis['detected_type'] ?? 'unknown') . '</strong></div></div></div>';
        $html .= '<div class="col-md-4 mb-2"><div class="border rounded p-2 h-100"><div class="text-muted f-12 text-uppercase">Recommended path</div><div><strong>' . e(strtoupper(str_replace('-', ' ', $recommendedMode))) . '</strong></div></div></div>';
        $html .= '<div class="col-md-4 mb-2"><div class="border rounded p-2 h-100"><div class="text-muted f-12 text-uppercase">Current Worksuite</div><div><strong>' . e($analysis['package_summary']['current_worksuite_version'] ?? 'unknown') . '</strong></div></div></div>';
        $html .= '<div class="col-md-4 mb-2"><div class="border rounded p-2 h-100"><div class="text-muted f-12 text-uppercase">Check summary</div><div><strong>' . e((string) $checkSummary['pass']) . '</strong> pass / <strong>' . e((string) $checkSummary['warn']) . '</strong> warn / <strong>' . e((string) $checkSummary['fail']) . '</strong> fail</div></div></div>';
        $html .= '</div>';

        if (!empty($analysis['package_summary'])) {
            $html .= '<div class="alert alert-light border"><strong>Package summary</strong><ul class="mb-0 mt-2">';
            foreach ($analysis['package_summary'] as $key => $value) {
                if ($value === null || $value === '') {
                    continue;
                }
                $label = ucwords(str_replace('_', ' ', (string) $key));
                $html .= '<li><strong>' . e($label) . ':</strong> ' . e((string) $value) . '</li>';
            }
            $html .= '</ul></div>';
        }


        if (!empty($analysis['risk_profile']) && is_array($analysis['risk_profile'])) {
            $risk = $analysis['risk_profile'];
            $html .= '<div class="alert alert-light border"><strong>Risk profile</strong><ul class="mb-0 mt-2">';
            $html .= '<li><strong>Workspace profile:</strong> ' . e($risk['workspace'] ?? 'worksuite') . '</li>';
            $html .= '<li><strong>Repair level:</strong> ' . e($risk['recommended_repair_level'] ?? 'safe') . '</li>';
            $html .= '<li><strong>Automatic fixes available:</strong> ' . e((string) ($risk['auto_fixable_count'] ?? 0)) . '</li>';
            $html .= '<li><strong>Protected paths preserved:</strong> ' . e(implode(', ', $risk['protected_paths'] ?? [])) . '</li>';
            $html .= '</ul></div>';
        }

        if (!empty($analysis['repair_plan']) && is_array($analysis['repair_plan'])) {
            $html .= '<div class="alert alert-light border"><strong>Repair plan</strong><ul class="mb-0 mt-2">';
            foreach ($analysis['repair_plan'] as $repair) {
                $html .= '<li><strong>' . e($repair['title'] ?? 'Repair') . ':</strong> ' . e($repair['detail'] ?? '');
                if (!empty($repair['mode'])) {
                    $html .= ' <span class="badge badge-light border ml-1">' . e(strtoupper((string) $repair['mode'])) . '</span>';
                }
                $html .= '</li>';
            }
            $html .= '</ul></div>';
        }

        if (!empty($analysis['auto_repairs']) && is_array($analysis['auto_repairs'])) {
            $html .= '<div class="alert alert-info"><strong>Automatic repairs applied or queued</strong><ul class="mb-0 mt-2">';
            foreach ($analysis['auto_repairs'] as $repair) {
                $html .= '<li><strong>' . e($repair['title'] ?? 'Repair') . ':</strong> ' . e($repair['detail'] ?? '');
                if (!empty($repair['target'])) {
                    $html .= ' <span class="d-block small mt-1">' . e($repair['target']) . '</span>';
                }
                $html .= '</li>';
            }
            $html .= '</ul></div>';
        }

        if (!empty($analysis['blocking_issues'])) {
            $html .= '<div class="alert alert-danger"><strong>Blocking issues</strong><ul class="mb-0 mt-2">';
            foreach ($analysis['blocking_issues'] as $issue) {
                $html .= '<li>' . $issue . '</li>';
            }
            $html .= '</ul></div>';
        }

        if (!empty($analysis['warnings'])) {
            $html .= '<div class="alert alert-warning"><strong>Warnings</strong><ul class="mb-0 mt-2">';
            foreach ($analysis['warnings'] as $warning) {
                $html .= '<li>' . $warning . '</li>';
            }
            $html .= '</ul></div>';
        }

        if (!empty($analysis['checks'])) {
            $html .= '<div class="table-responsive"><table class="table table-bordered table-sm mb-3"><thead><tr><th>Check</th><th>Status</th><th>Detail</th></tr></thead><tbody>';
            foreach ($analysis['checks'] as $check) {
                $badgeClass = $check['status'] === 'pass' ? 'success' : ($check['status'] === 'warn' ? 'warning' : 'danger');
                $html .= '<tr><td>' . e($check['label']) . '</td><td><span class="badge badge-' . $badgeClass . '">' . e(strtoupper($check['status'])) . '</span></td><td>' . e($check['detail']) . '</td></tr>';
            }
            $html .= '</tbody></table></div>';
        }

        if (!empty($analysis['suggested_action'])) {
            $html .= '<div class="alert alert-info"><strong>Suggested action:</strong> ' . e($analysis['suggested_action']) . '</div>';
        }


        if (!empty($analysis['visibility_report']) && is_array($analysis['visibility_report'])) {
            $visibility = $analysis['visibility_report'];
            $final = $visibility['final'] ?? [];
            $badgeClass = ($final['status'] ?? 'warn') === 'pass' ? 'success' : (($final['status'] ?? 'warn') === 'fail' ? 'danger' : 'warning');
            $html .= '<div class="alert alert-' . $badgeClass . '"><strong>User menu visibility</strong><div class="mt-2">' . e($final['detail'] ?? 'Visibility not evaluated.') . '</div></div>';
            $html .= '<div class="row">';
            foreach (['sidebar' => 'Sidebar readiness', 'routes' => 'Route resolution', 'permissions' => 'Permission chain', 'package_bridge' => 'Package bridge'] as $key => $label) {
                $block = $visibility[$key] ?? [];
                $html .= '<div class="col-md-6 mb-2"><div class="alert alert-light border h-100"><strong>' . e($label) . '</strong><div class="small mt-2">' . e($block['detail'] ?? 'No data') . '</div></div></div>';
            }
            $html .= '</div>';
            if (!empty($visibility['sidebar']['route_resolution']['missing_route_names'] ?? [])) {
                $html .= '<div class="alert alert-warning"><strong>Unresolved sidebar route names</strong><div class="small mt-2">' . e(implode(', ', $visibility['sidebar']['route_resolution']['missing_route_names'])) . '</div></div>';
            }
            if (!empty($visibility['permissions']['db_audit']['missing'] ?? [])) {
                $html .= '<div class="alert alert-warning"><strong>Missing seeded permissions</strong><div class="small mt-2">' . e(implode(', ', $visibility['permissions']['db_audit']['missing'])) . '</div></div>';
            }
            if (!empty($visibility['package_bridge']['db_entitlements']['packages'] ?? [])) {
                $html .= '<div class="alert alert-light border"><strong>Package entitlement rows</strong><div class="small mt-2">' . e(count($visibility['package_bridge']['db_entitlements']['packages'])) . ' package(s) currently include this module.</div></div>';
            }
            if (!empty($visibility['tenant']) && is_array($visibility['tenant'])) {
                $html .= '<div class="alert alert-light border"><strong>Tenant visibility simulation</strong><ul class="mb-0 mt-2">';
                foreach ($visibility['tenant'] as $profile => $result) {
                    $html .= '<li><strong>' . e(ucwords(str_replace('_', ' ', (string) $profile))) . ':</strong> ' . e(strtoupper((string) ($result['status'] ?? 'warn'))) . ' — ' . e($result['detail'] ?? '') . '</li>';
                }
                $html .= '</ul></div>';
            }
        }

        if (!empty($analysis['visibility_report']['checklist']) && is_array($analysis['visibility_report']['checklist'])) {
            $html .= '<div class="alert alert-light border"><strong>User menu visibility checklist</strong><ul class="mb-0 mt-2">';
            foreach ($analysis['visibility_report']['checklist'] as $item) {
                $html .= '<li><strong>' . e($item['label'] ?? 'Check') . ':</strong> ' . e(strtoupper((string) ($item['status'] ?? 'warn'))) . '</li>';
            }
            $html .= '</ul></div>';
        }

        if (!empty($analysis['repair_queue']) && is_array($analysis['repair_queue'])) {
            $html .= '<div class="alert alert-info"><strong>Queued repairs for visibility blockers</strong><ul class="mb-0 mt-2">';
            foreach ($analysis['repair_queue'] as $repair) {
                $html .= '<li><strong>' . e($repair['title'] ?? 'Repair') . ':</strong> ' . e($repair['detail'] ?? '') . ' <span class="badge badge-light border ml-1">' . e(strtoupper((string) ($repair['mode'] ?? 'safe'))) . '</span></li>';
            }
            $html .= '</ul></div>';
        }

        if (!empty($analysis['doctor_manifest']) && is_array($analysis['doctor_manifest'])) {
            $html .= '<div class="alert alert-light border"><strong>Doctor manifest</strong><div class="small mt-2">Workspace: ' . e($analysis['doctor_manifest']['workspace'] ?? 'worksuite') . '; features: ' . e(implode(', ', $analysis['doctor_manifest']['features'] ?? [])) . '</div></div>';
        }

        if (!empty($analysis['post_install']) && is_array($analysis['post_install'])) {
            $html .= '<div class="alert alert-light border"><strong>Post-install actions</strong><ul class="mb-0 mt-2">';
            foreach ($analysis['post_install'] as $label => $result) {
                $html .= '<li><strong>' . e(ucwords(str_replace('_', ' ', $label))) . ':</strong> ' . e(($result['status'] ?? 'info')) . ' — ' . e(($result['detail'] ?? ''));
                if (!empty($result['audit']) && is_array($result['audit'])) {
                    $html .= '<ul class="mt-1">';
                    foreach ($result['audit'] as $auditLabel => $auditValue) {
                        if (is_bool($auditValue)) {
                            $auditValue = $auditValue ? 'yes' : 'no';
                        }
                        $html .= '<li><strong>' . e(ucwords(str_replace('_', ' ', $auditLabel))) . ':</strong> ' . e((string) $auditValue) . '</li>';
                    }
                    $html .= '</ul>';
                }
                $html .= '</li>';
            }
            $html .= '</ul></div>';
        }

        $nextSteps = $this->buildNextStepChecklist($analysis);
        if (count($nextSteps)) {
            $html .= '<div class="alert alert-light border"><strong>Next steps</strong><ol class="mb-0 mt-2">';
            foreach ($nextSteps as $step) {
                $html .= '<li>' . e($step) . '</li>';
            }
            $html .= '</ol></div>';
        }

        $downloadName = ($analysis['module_name'] ?? 'module') . '-analysis.json';
        $jsonPayload = e(json_encode($analysis, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES));
        $html .= '<div class="d-flex flex-wrap gap-2 align-items-center">';
        $html .= '<button type="button" class="btn btn-outline-secondary btn-sm mr-2 copy-analysis-json" data-json-report="' . $jsonPayload . '">Copy JSON report</button>';
        $html .= '<button type="button" class="btn btn-outline-primary btn-sm download-analysis-json" data-download-name="' . e($downloadName) . '" data-json-report="' . $jsonPayload . '">Download JSON report</button>';
        if (!empty($analysis['log_id'])) {
            $html .= '<span class="badge badge-light border">Log #' . e((string) $analysis['log_id']) . '</span>';
        }
        $html .= '</div>';

        $html .= '</div>';

        return $html;
    }

    public function rollback(ModuleInstallLog $install, Request $request)
    {
        abort_403(GlobalSetting::validateSuperAdmin('manage_superadmin_custom_module_settings'));

        if (!$install->can_rollback || empty($install->pre_install_snapshot)) {
            return Reply::dataOnly([
                'status' => 'fail',
                'message' => 'Rollback not available for this installation.',
            ]);
        }

        $moduleName = $install->module_name;
        $results = [];

        try {
            if (class_exists(InstallationEventDispatcher::class)) {
                InstallationEventDispatcher::rollbackStarted($moduleName, ['install_id' => $install->id]);
            }
            $modulePath = base_path("Modules/{$moduleName}");
            if (File::exists($modulePath)) {
                File::deleteDirectory($modulePath);
                $results['files_deleted'] = true;
            }

            $restoreResult = app(ModuleInstallSnapshot::class)->restoreFrom($moduleName, $install->pre_install_snapshot ?? []);
            $results['snapshot_restore'] = $restoreResult;

            if (!empty($install->applied_repairs)) {
                $results['repair_rollback'] = app(SafeFixActionExecutor::class)->rollbackRepairs($moduleName, array_fill_keys($install->applied_repairs, true));
            }

            Artisan::call('cache:clear');
            Artisan::call('route:clear');
            Artisan::call('view:clear');
            Artisan::call('config:clear');
            cache()->forget('laravel-modules');
            File::deleteDirectory(storage_path('app') . '/Modules/');

            $install->update([
                'status' => 'rolled_back',
                'last_verified_at' => now(),
            ]);
            if (class_exists(InstallationEventDispatcher::class)) {
                InstallationEventDispatcher::rollbackCompleted($moduleName, $results);
            }

            return Reply::dataOnly([
                'status' => 'success',
                'message' => "Module {$moduleName} rolled back successfully.",
                'details' => $results,
            ]);
        } catch (\Throwable $e) {
            if (class_exists(InstallationEventDispatcher::class)) {
                InstallationEventDispatcher::rollbackFailed($moduleName, ['message' => $e->getMessage()]);
            }
            $install->update([
                'status' => 'rollback_failed',
                'installation_notes' => trim(($install->installation_notes ?? '') . ' Rollback failed: ' . $e->getMessage()),
                'last_verified_at' => now(),
            ]);

            return Reply::dataOnly([
                'status' => 'fail',
                'message' => 'Rollback failed: ' . $e->getMessage(),
            ]);
        }
    }

    public function analyzeDownload(Request $request)
    {
        return $this->downloadAnalysis($request);
    }

    public function downloadAnalysis(Request $request)
    {
        $analysis = $this->analyzeUploadedModule($request->filePath, false, $request->boolean('auto_repair_metadata', true));
        $analysis = $this->applyAdvancedValidatorSignals($analysis, $request->filePath);
        $log = $this->persistInstallLog('analysis_download', $analysis, $request->filePath);
        $analysis['log_id'] = $log?->id;

        return response()->json($analysis, ($analysis['status'] ?? false) ? 200 : 422, [
            'Content-Disposition' => 'attachment; filename="' . ($analysis['module_name'] ?? 'module') . '-analysis.json"',
        ], JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES);
    }

    private function applyAdvancedValidatorSignals(array $analysis, ?string $filePath, ?array $securityResult = null, ?array $permResult = null, ?array $routeResult = null): array
    {
        if (blank($filePath) || !File::exists($filePath)) {
            $analysis['diagnostics_html'] = $this->renderDiagnosticsHtml($analysis);
            return $analysis;
        }

        $securityResult ??= app(SecurityScanValidator::class)->validate($filePath);
        $permResult ??= app(PermissionCollisionValidator::class)->validate($filePath);
        $routeResult ??= app(RouteCollisionValidator::class)->validate($filePath);

        $analysis['package_summary']['checksum_sha256'] = $securityResult['checksum_sha256'] ?? ($analysis['package_summary']['checksum_sha256'] ?? null);
        $analysis['checks'][] = [
            'label' => 'Security scan',
            'status' => ($securityResult['valid'] ?? false) ? 'pass' : 'fail',
            'detail' => ($securityResult['valid'] ?? false)
                ? 'No dangerous shell or traversal patterns detected'
                : collect($securityResult['issues'] ?? [])->pluck('code')->filter()->implode(', '),
        ];
        $analysis['checks'][] = [
            'label' => 'Permission collision check',
            'status' => ($permResult['valid'] ?? false) ? 'pass' : 'fail',
            'detail' => (($permResult['permissions_count'] ?? 0) > 0)
                ? (($permResult['collisions_found'] ?? 0) . ' collision(s) across ' . ($permResult['permissions_count'] ?? 0) . ' permission(s)')
                : 'No declared permissions in manifest',
        ];
        $analysis['checks'][] = [
            'label' => 'Route collision check',
            'status' => ($routeResult['valid'] ?? false) ? 'pass' : 'fail',
            'detail' => (($routeResult['routes_count'] ?? 0) > 0)
                ? (($routeResult['collisions_found'] ?? 0) . ' collision(s) across ' . ($routeResult['routes_count'] ?? 0) . ' route(s)')
                : 'No route collisions detected',
        ];

        foreach ([['result' => $securityResult, 'defaultSeverity' => 'error'], ['result' => $permResult, 'defaultSeverity' => 'error'], ['result' => $routeResult, 'defaultSeverity' => 'error']] as $bundle) {
            foreach (($bundle['result']['issues'] ?? []) as $issue) {
                $message = $issue['message'] ?? null;
                if (!$message) {
                    continue;
                }
                if (($issue['severity'] ?? $bundle['defaultSeverity']) === 'warning') {
                    $analysis['warnings'][] = $message;
                } else {
                    $analysis['blocking_issues'][] = $message;
                }
            }
        }

        $analysis['warnings'] = array_values(array_unique($analysis['warnings'] ?? []));
        $analysis['blocking_issues'] = array_values(array_unique($analysis['blocking_issues'] ?? []));

        if (!empty($analysis['blocking_issues'])) {
            $analysis['status'] = false;
            $analysis['installability_status'] = 'blocked';
            $analysis['recommended_mode'] = 'repair';
            $analysis['suggested_action'] = 'Fix the blocking issues below, then upload the corrected module ZIP again.';
            $analysis['message'] = $analysis['message'] ?: 'The uploaded package has blocking validation issues.';
        }

        $analysis['diagnostics_html'] = $this->renderDiagnosticsHtml($analysis);
        return $analysis;
    }


    private function enrichRegistrySignals(array $analysis, ?string $filePath = null): array
    {
        if (!$filePath || !config('custom-modules.registry.enabled', true)) {
            return $analysis;
        }

        try {
            if (config('custom-modules.registry.auto_ingest_on_analyze', false) && class_exists(RemoteRegistryIngestor::class)) {
                app(RemoteRegistryIngestor::class)->ingestActiveFeeds();
            }

            if (class_exists(VersionResolver::class)) {
                $versionSignals = app(VersionResolver::class)->resolveForZip($filePath);
                $analysis['registry_signals'] = $versionSignals;
                if (!empty($versionSignals['module_name'])) {
                    $analysis['package_summary']['registry_module'] = $versionSignals['module_name'];
                }
                if (!empty($versionSignals['latest_version'])) {
                    $analysis['package_summary']['registry_latest_version'] = $versionSignals['latest_version'];
                }
                $analysis['checks'][] = [
                    'label' => 'Registry version resolution',
                    'status' => ($versionSignals['status'] ?? 'unknown') === 'upgrade_available' ? 'warn' : 'info',
                    'detail' => $versionSignals['message'] ?? (($versionSignals['latest_version'] ?? null)
                        ? 'Current ' . ($versionSignals['current_version'] ?? '0.0.0') . ' | Latest ' . ($versionSignals['latest_version'] ?? '0.0.0')
                        : 'No registry match found.'),
                ];
                if (($versionSignals['status'] ?? null) === 'upgrade_available') {
                    $analysis['warnings'][] = 'A newer registry version is available: ' . ($versionSignals['latest_version'] ?? 'unknown') . '.';
                }
            }
        } catch (\Throwable $e) {
            $analysis['warnings'][] = 'Registry resolution failed: ' . $e->getMessage();
        }

        return $analysis;
    }

    private function persistInstallLog(string $event, array $analysis, ?string $filePath = null)
    {
        try {
            if (!Schema::hasTable('module_install_logs')) {
                return null;
            }

            return ModuleInstallLog::create([
                'event' => $event,
                'module_name' => $analysis['module_name'] ?? null,
                'uploaded_filename' => $filePath ? basename($filePath) : null,
                'detected_type' => $analysis['detected_type'] ?? 'unknown',
                'installability_status' => $analysis['installability_status'] ?? 'blocked',
                'recommended_mode' => $analysis['recommended_mode'] ?? 'repair',
                'blocking_issues' => $analysis['blocking_issues'] ?? [],
                'warnings' => $analysis['warnings'] ?? [],
                'checks' => $analysis['checks'] ?? [],
                'package_summary' => $analysis['package_summary'] ?? [],
                'suggested_action' => $analysis['suggested_action'] ?? null,
                'created_by' => auth()->id(),
            ]);
        }
        catch (\Throwable $e) {
            logger()->warning('Module installer log write failed: ' . $e->getMessage());
            return null;
        }
    }

    private function buildNextStepChecklist(array $analysis): array
    {
        $steps = [];
        $status = $analysis['installability_status'] ?? 'blocked';
        $recommendedMode = $analysis['recommended_mode'] ?? 'repair';

        if ($status === 'ready') {
            $steps[] = 'Install the module from this screen.';
            $steps[] = 'Run migrations and clear caches after install.';
        }

        if ($status === 'installable-with-warnings') {
            $steps[] = 'Install only if the warnings are acceptable.';
            $steps[] = 'Review menu wiring, package access, and permissions immediately after install.';
        }

        if ($status === 'upgrade-candidate') {
            $steps[] = 'Back up the existing module folder before replacing it.';
            $steps[] = 'Treat this ZIP as an upgrade, not a first install.';
        }

        if ($status === 'overlay-only' || $recommendedMode === 'overlay') {
            $steps[] = 'Do not install this ZIP through the native installer.';
            $steps[] = 'Stage it as donor/overlay code and merge manually into the base app.';
        }

        if ($status === 'blocked') {
            $steps[] = 'Fix the blocking checks shown above.';
            $steps[] = 'Run Analyze ZIP again before trying to install.';
        }

        foreach ($analysis['checks'] ?? [] as $check) {
            if (($check['label'] ?? '') === 'Menu readiness' && ($check['status'] ?? '') === 'warn') {
                $steps[] = 'Add menu wiring or sidebar integration so the module is visible after install.';
            }
            if (($check['label'] ?? '') === 'Permission readiness' && ($check['status'] ?? '') === 'warn') {
                $steps[] = 'Add permission seeders or role mappings so users can access the module.';
            }
            if (($check['label'] ?? '') === 'Package bridge readiness' && ($check['status'] ?? '') === 'warn') {
                $steps[] = 'Add package/module bridge logic so the module is assignable to customer plans.';
            }
        }

        return array_values(array_unique($steps));
    }

    private function renderCheckSummary(array $checks)
    {
        return [
            'pass' => count(array_filter($checks, fn ($check) => ($check['status'] ?? null) === 'pass')),
            'warn' => count(array_filter($checks, fn ($check) => ($check['status'] ?? null) === 'warn')),
            'fail' => count(array_filter($checks, fn ($check) => ($check['status'] ?? null) === 'fail')),
        ];
    }

    public function validateModule($moduleName)
    {
        $appName = str_replace('-new', '', config('froiden_envato.envato_product_name'));
        $wrongMessage = 'The zip that you are trying to install is not compatible with ' . $appName . ' version';

        // Check if PHP-ZIP extension is missing
        if (!extension_loaded('zip')) {
            return [
                'status' => false,
                'message' => '<b>PHP-ZIP</b> extension is missing on your server. Please install the extension.'
            ];
        }

        $configPath = storage_path('app') . '/Modules/' . $moduleName . '/Config/config.php';

        // Check if module configuration file exists
        if (!file_exists($configPath)) {
            return [
                'status' => false,
                'message' => $wrongMessage
            ];
        }

        $config = require_once $configPath;

        $config['parent_envato_id'] = $config['parent_envato_id'] ?? config('froiden_envato.envato_item_id');
        $config['parent_product_name'] = $config['parent_product_name'] ?? config('froiden_envato.envato_product_name');
        if (!isset($config['parent_min_version']) && File::exists(base_path('version.txt'))) {
            $config['parent_min_version'] = trim(File::get(base_path('version.txt')));
        }

        // Check if parent_envato_id is defined and matches the application's envato_id
        if (!isset($config['parent_envato_id']) || $config['parent_envato_id'] !== config('froiden_envato.envato_item_id')) {
            return [
                'status' => false,
                'message' => 'You are installing the wrong module for this product'
            ];
        }

        // Parent envato id is different from module envato id
        if ($config['parent_envato_id'] !== config('froiden_envato.envato_item_id')) {
            return [
                'status' => false,
                'message' => 'You are installing wrong module for this product'
            ];
        }

        // Check if parent_min_version is defined
        if (!isset($config['parent_min_version'])) {
            $errorMessage = App::environment('codecanyon') ? 'Please download and install the latest version of the module.' : 'Minimum version of <b>' . $appName . ' main application</b> is not defined in the Module.';

            return [
                'status' => false,
                'message' => $errorMessage
            ];
        }

        // Check if the application version is lower than the required minimum version
        if (version_compare(trim(File::get('version.txt')), $config['parent_min_version'], '<')) {
            return [
                'status' => false,
                'message' => 'Minimum version of <b>' . $appName . ' main application</b> should be greater than or equal to <b>' . $config['parent_min_version'] . '</b>. Your application version is <b>' . File::get('version.txt') . '</b>'
            ];
        }


        // Check if parent_product_name is defined and matches the application's product name
        if (!isset($config['parent_product_name']) || $config['parent_product_name'] !== config('froiden_envato.envato_product_name')) {
            return [
                'status' => false,
                'message' => $wrongMessage
            ];
        }

        return [
            'status' => true,
            'message' => 'Unzipped successfully'
        ];
    }

    private function flushData()
    {
        Artisan::call('optimize:clear');
        Artisan::call('view:clear');
        $user = auth()->id();
        // clear cache
        cache()->flush();
        // clear session
        session()->flush();
        auth()->logout();
        // login user
        auth()->loginUsingId($user);
    }

    /**
     * Display the specified resource.
     *
     * @param int $id
     * @return \Illuminate\Contracts\View\Factory|\Illuminate\View\View
     */
    public function show($id)
    {
        return $this->verifyModulePurchase($id);
    }

    public function update(Request $request, $moduleName)
    {
        /** @phpstan-ignore-next-line */
        $module = Module::findOrFail($moduleName);

        $status = $request->status;

        ModuleSetting::where('module_name', $moduleName)->delete();

        ($status == 'active') ? $module->enable() : $module->disable();

        event(new ModuleStatusChanged($moduleName, $status));

        // We are registering the module to run the commands
        $module->register();

        /** @phpstan-ignore-next-line */
        $plugins = \Nwidart\Modules\Facades\Module::allEnabled();

        if ($status == 'active') {
            $this->runModuleMigrateCommand($moduleName);

            // We will call the module function php artisan asset:activate, zoom:active , etc
            $this->runActivateCommand(strtolower($moduleName));
        }

        $this->flushData();
        // WORKSUITESAAS
        if (strtolower($moduleName) == 'subdomain' && ($status == 'active')) {
            \session(['subdomain_module_activated' => true]);
        }

        cache()->forget('user_modules');
        /** @phpstan-ignore-next-line */
        cache(['worksuite_plugins' => array_keys($plugins)]);


        if (strtolower($moduleName) == 'languagepack' && $status == 'active') {
            session(['languagepack_module_activated' => true]);
        }

        return Reply::redirect(route('custom-modules.index') . '?tab=custom', 'Status Changed. Reloading');
    }

    public function verifyingModulePurchase(Request $request)
    {
        $request->validate([
            'purchase_code' => 'required|max:80',
        ]);

        $module = $request->module;
        $purchaseCode = $request->purchase_code;

        return $this->modulePurchaseVerified($module, $purchaseCode);
    }

    /**
     * @throws \Exception
     */
    private function unzipCodecanyon($zip)
    {
        $codeCanyonPath = storage_path('app') . '/Modules/Codecanyon';
        $zip->extract($codeCanyonPath);
        $files = File::allfiles($codeCanyonPath);

        foreach ($files as $file) {

            if (str_contains($file->getRelativePathname(), '.zip')) {
                $filePath = $file->getRelativePathname();
                $zip = Zip::open($codeCanyonPath . '/' . $filePath);
                $zip->extract(storage_path('app') . '/Modules');

                return $this->getZipName($filePath);
            }
        }

        return false;
    }

    private function getZipName($filePath)
    {
        $array = explode('/', str_replace('\\', '/', $filePath));

        return end($array);
    }

    /**
     * @param $moduleName
     * This will update the version of on server
     */
    private function updateVersion($moduleName)
    {
        try {
            $config = require base_path() . '/Modules/' . $moduleName . '/Config/config.php';
            $setting = (new $config['setting'])::first();

            // When module migrations are not run

            if ($setting?->purchase_code) {
                $this->modulePurchaseVerified(strtolower($moduleName), $setting->purchase_code);
            }
        } catch (\Exception $e) {
            logger($e->getMessage());
        }
    }

    private function runModuleMigrateCommand($moduleName)
    {
        Artisan::call('module:migrate', [$moduleName, '--force' => true]);
    }

    private function runActivateCommand($moduleName)
    {
        $command = $moduleName . ':activate';

        $artisanCommands = \Artisan::all();

        if (array_has($artisanCommands, $command)) {
            Artisan::call($command);
        }
    }
}

// PASS 15 marker: menu truth engine and provider boot verifier wired.
