<?php
namespace App\Support\ModuleDoctor;
class MenuSqlPreviewService {
    public function preview(string $routeName, string $label): string {
        return "INSERT INTO menus (menu_name, route, parent_id, extension, bolt_menu, params) VALUES ('{$label}', '{$routeName}', 0, '1', 0, '[]');";
    }
}
