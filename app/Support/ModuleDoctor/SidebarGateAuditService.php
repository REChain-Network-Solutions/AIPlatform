<?php

namespace App\Support\ModuleDoctor;

use Illuminate\Support\Facades\File;

class SidebarGateAuditService
{
    public function inspect(array $context, array $sidebarFiles = []): array
    {
        $gates = [];
        $unsupported = [];

        foreach ($sidebarFiles as $file) {
            if (!File::exists($file)) {
                continue;
            }
            $content = File::get($file);
            $relative = str_replace(rtrim((string) ($context['module_path'] ?? ''), '/') . '/', '', $file);
            foreach (['module_enabled(', 'user_modules()', 'permission(', 'hasPermission(', 'can('] as $pattern) {
                if (str_contains($content, $pattern)) {
                    $gates[] = ['file' => $relative, 'pattern' => $pattern];
                }
            }
            foreach (['isAbleTo(', '@permission', 'permission:'] as $pattern) {
                if (str_contains($content, $pattern)) {
                    $unsupported[] = ['file' => $relative, 'pattern' => $pattern];
                }
            }
        }

        return [
            'status' => count($unsupported) ? 'warn' : (count($gates) ? 'pass' : 'warn'),
            'detail' => count($unsupported)
                ? 'Sidebar gating includes unsupported permission APIs.'
                : (count($gates)
                    ? 'Sidebar contains recognizable Worksuite-style visibility gates.'
                    : 'No explicit Worksuite sidebar gate was detected; the module may still rely on external menu insertion.'),
            'gates' => $gates,
            'unsupported_gates' => $unsupported,
        ];
    }
}
