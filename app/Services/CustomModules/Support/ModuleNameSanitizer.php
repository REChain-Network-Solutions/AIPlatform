<?php
namespace App\Services\CustomModules\Support;

class ModuleNameSanitizer
{
    public function sanitize(string $name): string
    {
        return preg_replace('/[^A-Za-z0-9_\-\.]/', '_', $name) ?: ('module_' . time() . '.zip');
    }
}
