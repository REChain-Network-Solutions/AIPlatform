<?php

namespace App\Support\ModuleDoctor;

use Illuminate\Support\Facades\File;

class ModuleManifestInspector
{
    public function inspect(string $modulePath): array
    {
        $moduleJson = File::exists($modulePath . '/module.json') ? json_decode(File::get($modulePath . '/module.json'), true) : [];
        $config = File::exists($modulePath . '/Config/config.php') ? include $modulePath . '/Config/config.php' : [];

        return [
            'module_json' => is_array($moduleJson) ? $moduleJson : [],
            'config' => is_array($config) ? $config : [],
        ];
    }
}
