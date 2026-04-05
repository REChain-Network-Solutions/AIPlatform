<?php

namespace App\Support\ModuleDoctor;

use Illuminate\Support\Facades\File;

class SidebarInclusionAuditService
{
    public function inspect(array $context, array $sidebarFiles = []): array
    {
        $modulePath = (string) ($context['module_path'] ?? '');
        $moduleName = (string) ($context['module_name'] ?? 'Module');
        $needleVariants = [
            strtolower($moduleName),
            strtolower(str_replace(' ', '-', $moduleName)),
            strtolower(str_replace(' ', '_', $moduleName)),
        ];

        $coreSidebarFiles = [
            base_path('resources/views/sections/sidebar.blade.php'),
            base_path('resources/views/sections/menu.blade.php'),
            base_path('resources/views/components/setting-sidebar.blade.php'),
        ];

        $hits = [];
        foreach ($coreSidebarFiles as $file) {
            if (!File::exists($file)) {
                continue;
            }
            $content = strtolower(File::get($file));
            foreach ($needleVariants as $variant) {
                if ($variant && str_contains($content, $variant)) {
                    $hits[] = str_replace(base_path() . '/', '', $file);
                    break;
                }
            }
        }

        return [
            'status' => empty($sidebarFiles) ? 'warn' : (count($hits) ? 'pass' : 'warn'),
            'detail' => empty($sidebarFiles)
                ? 'No sidebar/menu blades were found in the uploaded module.'
                : (count($hits)
                    ? 'Potential include chain detected in existing Worksuite sidebar files.'
                    : 'Module sidebar files exist, but no clear inclusion signal was found in common Worksuite sidebar entry points.'),
            'core_sidebar_hits' => array_values(array_unique($hits)),
        ];
    }
}
