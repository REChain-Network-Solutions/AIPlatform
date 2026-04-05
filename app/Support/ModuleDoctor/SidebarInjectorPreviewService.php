<?php
namespace App\Support\ModuleDoctor;
class SidebarInjectorPreviewService {
    public function preview(string $routeName, string $label): string {
        return "<x-sub-menu-item :link=\"route('{$routeName}')\" :text=\"__('{$label}')\" />";
    }
}
