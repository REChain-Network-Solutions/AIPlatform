<?php

namespace App\Support\ModuleDoctor;

class JsonExportService
{
    public function encode(array $analysis): string
    {
        return (string) json_encode($analysis, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES);
    }
}
