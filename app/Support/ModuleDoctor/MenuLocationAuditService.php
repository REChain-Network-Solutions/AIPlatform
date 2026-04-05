<?php

namespace App\Support\ModuleDoctor;

use Illuminate\Support\Facades\File;

class MenuLocationAuditService
{
    public function inspect(array $context, array $sidebarFiles = []): array
    {
        $sections = [];
        foreach ($sidebarFiles as $file) {
            $relative = str_replace(rtrim((string) ($context['module_path'] ?? ''), '/') . '/', '', $file);
            if (str_contains($relative, '/work/')) {
                $sections[] = 'work';
            }
            if (str_contains($relative, '/settings/')) {
                $sections[] = 'settings';
            }
            if (str_contains($relative, '/superadmin/')) {
                $sections[] = 'superadmin';
            }
            if (str_contains($relative, '/admin/')) {
                $sections[] = 'admin';
            }
        }

        $sections = array_values(array_unique($sections));

        return [
            'status' => count($sections) ? 'pass' : 'warn',
            'detail' => count($sections)
                ? 'Sidebar/menu files target these sections: ' . implode(', ', $sections) . '.'
                : 'No clear user workspace or settings section could be inferred from sidebar file locations.',
            'sections' => $sections,
        ];
    }
}
