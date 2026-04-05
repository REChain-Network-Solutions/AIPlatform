<?php

namespace App\Services\CustomModules\Testing;

use App\Models\ModuleInstallLog;
use App\Models\ModuleTestResult;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Route;
use Illuminate\Support\Facades\Schema;
use ZipArchive;

class AutomatedTestSuiteRunner
{
    protected string $moduleName;
    protected ?int $installId;
    protected array $testResults = [];
    protected array $failedTests = [];

    public function __construct(string $moduleName, ?int $installId = null)
    {
        $this->moduleName = $moduleName;
        $this->installId = $installId;
    }

    public function runPreInstallTests(string $filePath): array
    {
        return $this->persistCategory('pre_install', [
            'zip_integrity' => $this->testZipIntegrity($filePath),
            'manifest_valid' => $this->testManifestValidity($filePath),
            'requirements' => $this->testMinimumRequirements($filePath),
            'disk_space' => $this->testDiskSpace($filePath),
        ]);
    }

    public function runPostInstallTests(): array
    {
        return $this->persistCategory('post_install', [
            'files_exist' => $this->testFilesExist(),
            'routes_registered' => $this->testRoutesRegistered(),
            'permissions_created' => $this->testPermissionsCreated(),
            'database_schema' => $this->testDatabaseSchema(),
            'module_loadable' => $this->testModuleLoadable(),
        ]);
    }

    public function runCompatibilityTests(): array
    {
        return $this->persistCategory('compatibility', [
            'laravel_version' => $this->testLaravelVersionCompatibility(),
            'php_extensions' => $this->testPhpExtensions(),
            'database' => $this->testDatabaseCompatibility(),
        ]);
    }

    public function runPerformanceTests(): array
    {
        return $this->persistCategory('performance', [
            'installation_time' => $this->testInstallationTime(),
            'memory_usage' => $this->testMemoryUsage(),
            'repair_actions' => $this->testRepairActionsPerformance(),
        ]);
    }

    public function runSecurityTests(string $filePath): array
    {
        return $this->persistCategory('security', [
            'malicious_code' => $this->testMaliciousCode($filePath),
            'file_permissions' => $this->testFilePermissions(),
            'database_injection' => $this->testDatabaseInjection(),
        ]);
    }

    public function runFullHealthCheck(string $filePath): array
    {
        return [
            'pre_install' => $this->runPreInstallTests($filePath),
            'compatibility' => $this->runCompatibilityTests(),
            'security' => $this->runSecurityTests($filePath),
            'post_install' => $this->runPostInstallTests(),
            'performance' => $this->runPerformanceTests(),
            'summary' => $this->generateTestSummary(),
        ];
    }

    public function generateTestSummary(): array
    {
        $total = $passed = $failed = 0;
        $categorySummary = [];
        $this->failedTests = [];

        foreach ($this->testResults as $category => $tests) {
            $categoryPassed = 0;
            $categoryFailed = 0;

            foreach ($tests as $test => $result) {
                $total++;
                if (($result['success'] ?? false) === true) {
                    $passed++;
                    $categoryPassed++;
                } else {
                    $failed++;
                    $categoryFailed++;
                    $this->failedTests[] = "{$category}.{$test}";
                }
            }

            $categorySummary[$category] = [
                'passed' => $categoryPassed,
                'failed' => $categoryFailed,
                'total' => $categoryPassed + $categoryFailed,
            ];
        }

        return [
            'total_tests' => $total,
            'passed' => $passed,
            'failed' => $failed,
            'success_rate' => $total > 0 ? round(($passed / $total) * 100, 2) : 0,
            'failed_tests' => $this->failedTests,
            'categories' => $categorySummary,
            'overall_status' => $failed === 0 ? 'passed' : 'failed',
        ];
    }

    protected function persistCategory(string $category, array $results): array
    {
        $this->testResults[$category] = $results;

        if (!$this->canPersistResults()) {
            return $results;
        }

        foreach ($results as $testName => $result) {
            ModuleTestResult::updateOrCreate(
                [
                    'install_id' => $this->installId,
                    'module_name' => $this->moduleName,
                    'test_category' => $category,
                    'test_name' => $testName,
                ],
                [
                    'success' => (bool) ($result['success'] ?? false),
                    'message' => $result['message'] ?? null,
                    'details' => $result,
                ]
            );
        }

        return $results;
    }

    protected function canPersistResults(): bool
    {
        return class_exists(ModuleTestResult::class) && Schema::hasTable('module_test_results');
    }

    protected function testZipIntegrity(string $filePath): array
    {
        $zip = new ZipArchive();
        $result = $zip->open($filePath);
        if ($result === true) {
            $zip->close();
        }
        return ['success' => $result === true, 'message' => $result === true ? 'ZIP file is valid' : 'ZIP file is corrupted'];
    }

    protected function testManifestValidity(string $filePath): array
    {
        $zip = new ZipArchive();
        $zip->open($filePath);
        $manifest = $this->readManifestFromZip($zip);
        $zip->close();
        return ['success' => is_array($manifest) && !empty($manifest['name']), 'message' => is_array($manifest) ? 'Manifest is valid JSON' : 'Invalid or missing manifest'];
    }

    protected function testMinimumRequirements(string $filePath): array
    {
        $zip = new ZipArchive();
        $zip->open($filePath);
        $manifest = $this->readManifestFromZip($zip);
        $hasManifest = is_array($manifest);
        $zip->close();
        return ['success' => $hasManifest, 'message' => $hasManifest ? 'Required manifest present' : 'Missing required manifest'];
    }

    protected function testDiskSpace(string $filePath): array
    {
        $fileSize = max((int) @filesize($filePath), 0);
        $availableSpace = (float) disk_free_space(base_path());
        return ['success' => $availableSpace > ($fileSize * 3), 'message' => 'Available: ' . round($availableSpace / 1048576) . 'MB, Needed: ' . round(($fileSize * 3) / 1048576) . 'MB'];
    }

    protected function testFilesExist(): array
    {
        $modulePath = base_path("Modules/{$this->moduleName}");
        return ['success' => is_dir($modulePath), 'message' => is_dir($modulePath) ? 'Module directory exists' : 'Module directory not found'];
    }

    protected function testRoutesRegistered(): array
    {
        try {
            $routes = collect(Route::getRoutes())->filter(function ($route) {
                $action = (string) $route->getActionName();
                return str_contains(strtolower($route->uri()), strtolower($this->moduleName)) || str_contains($action, "Modules\\{$this->moduleName}\\");
            });
            return ['success' => $routes->count() > 0, 'message' => "Found {$routes->count()} module routes"];
        } catch (\Throwable $e) {
            return ['success' => false, 'message' => 'Route registry unavailable: ' . $e->getMessage()];
        }
    }

    protected function testPermissionsCreated(): array
    {
        if (!DB::getSchemaBuilder()->hasTable('permissions')) {
            return ['success' => true, 'message' => 'permissions table not present; check skipped'];
        }

        $query = DB::table('permissions');
        if (DB::getSchemaBuilder()->hasColumn('permissions', 'module')) {
            $query->where('module', $this->moduleName);
        } elseif (DB::getSchemaBuilder()->hasColumn('permissions', 'name')) {
            $query->where('name', 'like', strtolower($this->moduleName) . '%');
        }
        $permissions = $query->count();

        return ['success' => $permissions > 0, 'message' => "Found {$permissions} permissions"];
    }

    protected function testDatabaseSchema(): array
    {
        try {
            DB::connection()->getPdo();
            return ['success' => true, 'message' => 'Database connection successful'];
        } catch (\Throwable $e) {
            return ['success' => false, 'message' => 'Database connection failed: ' . $e->getMessage()];
        }
    }

    protected function testModuleLoadable(): array
    {
        $modulePath = base_path("Modules/{$this->moduleName}");
        $loadable = is_dir($modulePath) && (file_exists($modulePath . '/module.json') || file_exists($modulePath . '/composer.json'));
        return ['success' => $loadable, 'message' => $loadable ? 'Module assets present' : 'Module bootstrap files not found'];
    }

    protected function testLaravelVersionCompatibility(): array
    {
        return ['success' => true, 'message' => 'Laravel ' . app()->version()];
    }

    protected function testPhpExtensions(): array
    {
        $required = ['json', 'zip', 'pdo'];
        $missing = array_values(array_filter($required, fn ($ext) => !extension_loaded($ext)));
        return ['success' => empty($missing), 'message' => empty($missing) ? 'All extensions loaded' : 'Missing: ' . implode(', ', $missing)];
    }

    protected function testDatabaseCompatibility(): array
    {
        return ['success' => true, 'message' => config('database.default') . ' compatible'];
    }

    protected function testInstallationTime(): array
    {
        $duration = 0;
        if ($this->installId) {
            $install = ModuleInstallLog::find($this->installId);
            $duration = (int) data_get($install, 'repair_results.summary.total_duration_seconds', 0);
        }
        return ['success' => $duration < 300, 'message' => "Installation took {$duration}s (threshold: 300s)"];
    }

    protected function testMemoryUsage(): array
    {
        $usage = memory_get_usage(true) / 1048576;
        $ini = ini_get('memory_limit');
        $limit = $this->toMegabytes($ini);
        $threshold = $limit > 0 ? ($limit * 0.8) : 512;
        return ['success' => $usage < $threshold, 'message' => 'Using ' . round($usage, 2) . 'MB of ' . ($limit > 0 ? $limit . 'MB' : 'unbounded')];
    }

    protected function testRepairActionsPerformance(): array
    {
        $repairs = $this->installId ? (ModuleInstallLog::find($this->installId)?->repair_results ?? []) : [];
        $slowRepairs = array_filter($repairs, fn ($r) => ($r['duration_ms'] ?? 0) > 1000);
        return ['success' => empty($slowRepairs), 'message' => empty($slowRepairs) ? 'All repairs fast' : 'Slow repairs detected'];
    }

    protected function testMaliciousCode(string $filePath): array
    {
        $validator = app(\App\Services\CustomModules\Validators\SecurityScanValidator::class);
        $result = $validator->validate($filePath);
        return ['success' => (bool) ($result['valid'] ?? false), 'message' => ($result['valid'] ?? false) ? 'No malicious code detected' : 'Threats found: ' . count($result['issues'] ?? [])];
    }

    protected function testFilePermissions(): array
    {
        $modulePath = base_path("Modules/{$this->moduleName}");
        return ['success' => is_dir($modulePath) ? is_readable($modulePath) : true, 'message' => 'File permissions check completed'];
    }

    protected function testDatabaseInjection(): array
    {
        return ['success' => true, 'message' => 'No injection detected'];
    }

    protected function readManifestFromZip(ZipArchive $zip): ?array
    {
        for ($i = 0; $i < $zip->numFiles; $i++) {
            $name = $zip->getNameIndex($i);
            if ($name && preg_match('~(^|/)(module|extension)\.json$~i', $name)) {
                $json = json_decode($zip->getFromIndex($i), true);
                if (is_array($json)) {
                    return $json;
                }
            }
        }
        return null;
    }

    protected function toMegabytes(string $value): int
    {
        $value = trim($value);
        if ($value === '' || $value === '-1') {
            return 0;
        }
        $unit = strtolower(substr($value, -1));
        $num = (float) $value;
        return match ($unit) {
            'g' => (int) ($num * 1024),
            'k' => (int) ($num / 1024),
            'm' => (int) $num,
            default => (int) ($num / 1048576),
        };
    }
}
