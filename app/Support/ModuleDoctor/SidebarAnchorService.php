<?php
namespace App\Support\ModuleDoctor;

class SidebarAnchorService
{
    public function anchors(array $module = []): array
    {
        return $this->audit($module);
    }

    public function audit(array $module = []): array
    {
        return [];
    }
}
