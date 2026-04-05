<?php

namespace App\Support\ModuleDoctor;

class ProtectedPathService
{
    public function all(): array
    {
        return [
            'resources/views/sections/menu.blade.php',
            'app/Console/Commands/*',
        ];
    }
}
