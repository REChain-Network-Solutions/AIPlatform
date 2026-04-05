<?php
namespace App\Support\ModuleDoctor;

class RegisteredRouteCatalogService
{
    public function all(): array
    {
        return $this->routes();
    }

    public function routes(): array
    {
        return [];
    }
}
