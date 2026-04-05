<?php

namespace App\Support\ModuleDoctor;

use Illuminate\Support\Facades\File;

class MenuRegistrationAuditService
{
    public function inspect(array $context): array
    {
        $moduleName = strtolower((string) ($context['module_name'] ?? ''));
        $hits = [];
        $files = [
            base_path('resources/views/sections/sidebar.blade.php'),
            base_path('resources/views/sections/menu.blade.php'),
            base_path('routes/web-settings.php'),
            base_path('routes/SuperAdmin/web.php'),
        ];

        foreach ($files as $file) {
            if (!File::exists($file)) {
                continue;
            }
            $content = strtolower(File::get($file));
            if ($moduleName && str_contains($content, $moduleName)) {
                $hits[] = str_replace(base_path() . '/', '', $file);
            }
        }

        return [
            'status' => count($hits) ? 'pass' : 'warn',
            'detail' => count($hits) ? 'Detected module references in live Worksuite menu/sidebar entry files.' : 'No live menu/sidebar registration references were detected in common Worksuite entry files.',
            'hits' => array_values(array_unique($hits)),
        ];
    }
}
