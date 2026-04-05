<?php

namespace App\Support\ModuleDoctor;

use Illuminate\Support\Facades\File;

class RouteBootPathAuditService
{
    public function inspect(array $context): array
    {
        $modulePath = (string) ($context['module_path'] ?? '');
        $providerFiles = [
            $modulePath . '/Providers/RouteServiceProvider.php',
            $modulePath . '/Providers/' . ($context['module_name'] ?? 'Module') . 'ServiceProvider.php',
        ];

        $hits = [];
        foreach ($providerFiles as $file) {
            if (File::exists($file)) {
                $content = File::get($file);
                if (str_contains($content, 'loadRoutesFrom') || str_contains($content, 'mapWebRoutes') || str_contains($content, 'Routes/web.php') || str_contains($content, 'Routes/api.php')) {
                    $hits[] = str_replace($modulePath . '/', '', $file);
                }
            }
        }

        return [
            'status' => count($hits) ? 'pass' : 'warn',
            'detail' => count($hits)
                ? 'Route boot path was detected in provider code.'
                : 'No obvious route boot path was detected in provider files.',
            'provider_hits' => $hits,
        ];
    }
}
