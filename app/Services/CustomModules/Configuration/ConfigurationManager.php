<?php

namespace App\Services\CustomModules\Configuration;

use App\Models\ModuleConfiguration;
use Illuminate\Support\Facades\Cache;
use Illuminate\Support\Facades\File;

class ConfigurationManager
{
    protected const CACHE_KEY = 'module_installer_config';
    protected const CACHE_TTL = 3600;

    public static function load(string $environment = null): array
    {
        $environment = $environment ?? app()->environment();
        $cacheKey = self::CACHE_KEY . ':' . $environment;

        return Cache::remember($cacheKey, self::CACHE_TTL, function () use ($environment) {
            $config = self::getDefaultConfig();

            if (class_exists(ModuleConfiguration::class) && self::configurationTableExists()) {
                $records = ModuleConfiguration::query()
                    ->whereIn('environment', [$environment, 'global'])
                    ->orderByRaw("case when environment = ? then 0 else 1 end", [$environment])
                    ->get();

                foreach ($records as $record) {
                    data_set($config, $record->key, self::normaliseValue($record->value), true);
                }
            }

            $configPath = config_path('module-installer');
            if (File::isDirectory($configPath)) {
                foreach (self::loadConfigFiles($configPath) as $fileConfig) {
                    $config = array_replace_recursive($config, $fileConfig);
                }
            }

            return $config;
        });
    }

    public static function save(string $key, $value, string $environment = null): ModuleConfiguration
    {
        $environment = $environment ?? app()->environment();

        $config = ModuleConfiguration::updateOrCreate(
            ['key' => $key, 'environment' => $environment],
            ['value' => is_array($value) ? json_encode($value) : (string) $value]
        );

        Cache::forget(self::CACHE_KEY . ':' . $environment);
        return $config;
    }

    public static function get(string $key, $default = null)
    {
        return data_get(self::load(), $key, $default);
    }

    public static function getValidatorConfig(string $validatorName): array
    {
        return self::get("validators.{$validatorName}", self::getDefaultValidatorConfig($validatorName));
    }

    public static function getRepairActionConfig(string $actionName): array
    {
        return self::get("repairs.{$actionName}", ['enabled' => true, 'timeout_seconds' => 30, 'retry_attempts' => 1]);
    }

    public static function getPerformanceConfig(): array
    {
        return self::get('performance', self::getDefaultConfig()['performance']);
    }

    public static function getSecurityConfig(): array
    {
        return self::get('security', self::getDefaultConfig()['security']);
    }

    protected static function getDefaultConfig(): array
    {
        return [
            'validators' => [
                'security' => ['enabled' => true, 'timeout_seconds' => 30, 'scan_level' => 'strict'],
                'permissions' => ['enabled' => true, 'allow_duplicates' => false],
                'routes' => ['enabled' => true, 'allow_duplicates' => false],
            ],
            'repairs' => [
                'permissions' => ['enabled' => true, 'timeout_seconds' => 30],
                'package_bridge' => ['enabled' => true, 'timeout_seconds' => 30],
                'company_settings' => ['enabled' => true, 'timeout_seconds' => 30],
                'cache_clear' => ['enabled' => true, 'timeout_seconds' => 10],
            ],
            'performance' => ['max_installation_time_seconds' => 300, 'slow_operation_threshold_ms' => 1000, 'memory_limit_mb' => 256],
            'security' => ['require_ssl' => false, 'ip_whitelist' => [], 'api_rate_limit' => 100, 'scan_archives' => true, 'block_dangerous_functions' => true],
            'retention' => ['keep_logs_days' => 90, 'keep_snapshots_days' => 30, 'keep_backups_days' => 60],
        ];
    }

    protected static function getDefaultValidatorConfig(string $validatorName): array
    {
        return self::getDefaultConfig()['validators'][$validatorName] ?? [];
    }

    protected static function loadConfigFiles(string $path): array
    {
        $configs = [];
        foreach (File::files($path) as $file) {
            $extension = strtolower($file->getExtension());
            $contents = File::get($file->getRealPath());
            $parsed = null;
            if ($extension === 'json') {
                $parsed = json_decode($contents, true);
            } elseif (in_array($extension, ['yaml', 'yml'], true) && class_exists(\Symfony\Component\Yaml\Yaml::class)) {
                $parsed = \Symfony\Component\Yaml\Yaml::parse($contents);
            }
            if (is_array($parsed)) {
                $configs[] = $parsed;
            }
        }
        return $configs;
    }

    public static function export(string $environment = null): array
    {
        $environment = $environment ?? app()->environment();
        return ModuleConfiguration::whereIn('environment', [$environment, 'global'])->get()->toArray();
    }

    public static function import(array $configs, string $environment = null): void
    {
        $environment = $environment ?? app()->environment();
        foreach ($configs as $config) {
            ModuleConfiguration::updateOrCreate(
                ['key' => $config['key'], 'environment' => $environment],
                ['value' => is_array($config['value'] ?? null) ? json_encode($config['value']) : ($config['value'] ?? null)]
            );
        }
        Cache::forget(self::CACHE_KEY . ':' . $environment);
    }

    public static function resetToDefaults(string $environment = null): void
    {
        $environment = $environment ?? app()->environment();
        ModuleConfiguration::where('environment', $environment)->delete();
        Cache::forget(self::CACHE_KEY . ':' . $environment);
    }

    protected static function normaliseValue($value)
    {
        if (!is_string($value)) {
            return $value;
        }
        $trimmed = trim($value);
        if ($trimmed === '') {
            return $value;
        }
        if (in_array(strtolower($trimmed), ['true', 'false'], true)) {
            return strtolower($trimmed) === 'true';
        }
        if (is_numeric($trimmed)) {
            return str_contains($trimmed, '.') ? (float) $trimmed : (int) $trimmed;
        }
        if ((str_starts_with($trimmed, '{') && str_ends_with($trimmed, '}')) || (str_starts_with($trimmed, '[') && str_ends_with($trimmed, ']'))) {
            $json = json_decode($trimmed, true);
            if (json_last_error() === JSON_ERROR_NONE) {
                return $json;
            }
        }
        return $value;
    }

    protected static function configurationTableExists(): bool
    {
        try {
            return \Illuminate\Support\Facades\Schema::hasTable('module_configurations');
        } catch (\Throwable $e) {
            return false;
        }
    }
}
