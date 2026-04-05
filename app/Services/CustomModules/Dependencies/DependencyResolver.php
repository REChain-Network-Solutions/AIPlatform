<?php

namespace App\Services\CustomModules\Dependencies;

use App\Models\ModuleDependency;
use Illuminate\Support\Facades\Schema;
use ZipArchive;

class DependencyResolver
{
    protected array $modules = [];
    protected array $versions = [];

    public function registerModule(
        string $name,
        string $version,
        array $dependencies = [],
        array $compatibleVersions = []
    ): void {
        $normalizedDependencies = $this->normalizeDependencies($dependencies);

        $this->modules[$name] = [
            'name' => $name,
            'version' => $version,
            'dependencies' => $normalizedDependencies,
            'compatible_with' => $compatibleVersions,
        ];

        $this->versions[$name] = $version;
        $this->persistDependencies($name, $normalizedDependencies);
    }

    public function extractDependenciesFromZip(string $zipPath): array
    {
        $zip = new ZipArchive();
        $result = $zip->open($zipPath);

        if ($result !== true) {
            return [];
        }

        $manifests = [];

        for ($i = 0; $i < $zip->numFiles; $i++) {
            $name = $zip->getNameIndex($i);
            if (!$name || !preg_match('~(^|/)(module|extension)\.json$~i', $name)) {
                continue;
            }

            $parsed = json_decode((string) $zip->getFromIndex($i), true);
            if (!is_array($parsed)) {
                continue;
            }

            $moduleName = $parsed['name'] ?? basename(dirname($name));
            $version = $parsed['version'] ?? '0.0.0';
            $dependencies = $this->normalizeDependencies($parsed['dependencies'] ?? []);
            $compatible = is_array($parsed['compatible_with'] ?? null) ? $parsed['compatible_with'] : [];

            $manifests[] = [
                'module' => $moduleName,
                'version' => $version,
                'dependencies' => $dependencies,
                'compatible_with' => $compatible,
                'manifest_path' => $name,
            ];

            $this->registerModule($moduleName, $version, $dependencies, $compatible);
        }

        $zip->close();

        if (count($manifests) === 1) {
            return $manifests[0];
        }

        return ['modules' => $manifests];
    }

    public function buildInstallPlanFromZip(string $zipPath): array
    {
        $extracted = $this->extractDependenciesFromZip($zipPath);
        $manifests = isset($extracted['modules']) ? $extracted['modules'] : [$extracted];
        $moduleNames = array_values(array_filter(array_map(fn ($row) => $row['module'] ?? null, $manifests)));
        $order = $this->resolveInstallationOrder($moduleNames);

        return [
            'modules' => $manifests,
            'installation_order' => $order,
            'dependency_graphs' => array_map(fn ($name) => $this->getDependencyGraph($name), $order),
            'unresolved_dependencies' => $this->findUnresolvedDependencies($moduleNames),
        ];
    }

    public function resolveInstallationOrder(array $modulesToInstall): array
    {
        $resolved = [];
        $visiting = [];

        foreach ($modulesToInstall as $module) {
            $this->topologicalSort($module, $resolved, $visiting);
        }

        return array_values(array_unique($resolved));
    }

    public function checkCircularDependencies(string $moduleName): array
    {
        $visited = [];
        $stack = [];

        return $this->detectCycle($moduleName, $visited, $stack);
    }

    public function checkVersionCompatibility(string $moduleName, string $requiredVersion): bool
    {
        $currentVersion = $this->versions[$moduleName] ?? null;

        if (!$currentVersion) {
            return false;
        }

        return $this->isVersionCompatible($currentVersion, $requiredVersion);
    }

    public function getDependencyGraph(string $moduleName): array
    {
        $graph = [
            'module' => $moduleName,
            'current_version' => $this->versions[$moduleName] ?? 'not_registered',
            'dependencies' => [],
        ];

        if (!isset($this->modules[$moduleName])) {
            return $graph;
        }

        foreach ($this->modules[$moduleName]['dependencies'] as $dep => $version) {
            $installed = isset($this->versions[$dep]);
            $graph['dependencies'][] = [
                'module' => $dep,
                'required_version' => $version,
                'current_version' => $this->versions[$dep] ?? 'not_installed',
                'compatible' => $installed ? $this->checkVersionCompatibility($dep, $version) : false,
                'status' => $installed ? ($this->checkVersionCompatibility($dep, $version) ? 'ok' : 'version_mismatch') : 'missing',
            ];
        }

        return $graph;
    }

    public function getCompatibilityMatrix(): array
    {
        $matrix = [];

        foreach ($this->modules as $moduleName => $module) {
            $matrix[$moduleName] = [
                'current_version' => $module['version'],
                'compatible_with' => $module['compatible_with'] ?? [],
                'depends_on' => $module['dependencies'] ?? [],
                'unresolved_count' => count(array_filter($this->getDependencyGraph($moduleName)['dependencies'], fn ($dep) => $dep['status'] !== 'ok')),
            ];
        }

        return $matrix;
    }

    public function resolveConflicts(array $requestedVersions): array
    {
        $resolved = [];
        $conflicts = [];

        foreach ($requestedVersions as $module => $version) {
            $current = $this->versions[$module] ?? null;
            if ($current && !$this->isVersionCompatible($current, $version)) {
                $conflicts[] = [
                    'module' => $module,
                    'current_version' => $current,
                    'requested_version' => $version,
                    'reason' => 'version_not_compatible',
                ];
                continue;
            }

            $resolved[$module] = $current ?: $version;
        }

        return ['resolved' => $resolved, 'conflicts' => $conflicts];
    }

    public function getAllDependencies(string $moduleName): array
    {
        $deps = [];
        $visited = [];
        $this->collectDependencies($moduleName, $deps, $visited);
        return $deps;
    }

    public function findUnresolvedDependencies(array $modulesToInstall): array
    {
        $unresolved = [];

        foreach ($modulesToInstall as $module) {
            foreach (($this->modules[$module]['dependencies'] ?? []) as $dep => $requiredVersion) {
                if (!isset($this->versions[$dep])) {
                    $unresolved[] = [
                        'module' => $module,
                        'missing_dependency' => $dep,
                        'required_version' => $requiredVersion,
                    ];
                }
            }
        }

        return $unresolved;
    }

    protected function topologicalSort(string $module, array &$resolved, array &$visiting): void
    {
        if (in_array($module, $resolved, true)) {
            return;
        }

        if (in_array($module, $visiting, true)) {
            throw new \RuntimeException("Circular dependency detected: {$module}");
        }

        $visiting[] = $module;

        foreach (($this->modules[$module]['dependencies'] ?? []) as $dep => $version) {
            if (!isset($this->modules[$dep])) {
                continue;
            }
            $this->topologicalSort($dep, $resolved, $visiting);
        }

        $resolved[] = $module;
        $visiting = array_values(array_diff($visiting, [$module]));
    }

    protected function detectCycle(string $node, array &$visited, array &$stack): array
    {
        $visited[$node] = true;
        $stack[] = $node;

        foreach (($this->modules[$node]['dependencies'] ?? []) as $neighbor => $version) {
            if (!isset($visited[$neighbor])) {
                $cycle = $this->detectCycle($neighbor, $visited, $stack);
                if (!empty($cycle)) {
                    return $cycle;
                }
            } elseif (in_array($neighbor, $stack, true)) {
                $cycleStart = array_search($neighbor, $stack, true);
                return array_slice($stack, (int) $cycleStart);
            }
        }

        array_pop($stack);
        return [];
    }

    protected function collectDependencies(string $moduleName, array &$deps, array &$visited): void
    {
        if (isset($visited[$moduleName])) {
            return;
        }

        $visited[$moduleName] = true;

        foreach (($this->modules[$moduleName]['dependencies'] ?? []) as $dep => $version) {
            $deps[$dep] = $version;
            $this->collectDependencies($dep, $deps, $visited);
        }
    }

    protected function normalizeDependencies(array $dependencies): array
    {
        $normalized = [];
        foreach ($dependencies as $key => $value) {
            if (is_int($key)) {
                $normalized[(string) $value] = '*';
            } else {
                $normalized[(string) $key] = is_string($value) && $value !== '' ? $value : '*';
            }
        }
        return $normalized;
    }

    protected function persistDependencies(string $moduleName, array $dependencies): void
    {
        if (!class_exists(ModuleDependency::class) || !Schema::hasTable('module_dependencies')) {
            return;
        }

        foreach ($dependencies as $dependency => $requiredVersion) {
            ModuleDependency::updateOrCreate(
                ['module_name' => $moduleName, 'depends_on' => $dependency],
                [
                    'required_version' => $requiredVersion,
                    'status' => isset($this->versions[$dependency]) && $this->isVersionCompatible($this->versions[$dependency], $requiredVersion) ? 'resolved' : 'pending',
                    'source' => 'runtime_registry',
                    'notes' => null,
                ]
            );
        }
    }

    protected function isVersionCompatible(string $current, string $required): bool
    {
        $required = trim($required);
        if ($required === '' || $required === '*' || $required === 'latest') {
            return true;
        }

        if (str_starts_with($required, '^')) {
            $base = ltrim($required, '^');
            $major = (int) explode('.', $base)[0];
            $currentMajor = (int) explode('.', $current)[0];
            return $currentMajor === $major && version_compare($current, $base, '>=');
        }

        if (str_starts_with($required, '>=')) {
            return version_compare($current, substr($required, 2), '>=');
        }

        if (preg_match('/^[0-9]+(\.[0-9]+){0,2}$/', $required)) {
            return version_compare($current, $required, '>=');
        }

        return version_compare($current, preg_replace('/[^0-9.]/', '', $required), '>=');
    }
}
