<?php

namespace App\Support\ModuleDoctor;

class RouteNameMismatchExplainerService
{
    public function explain(array $missingRouteNames = []): array
    {
        $items = collect($missingRouteNames)->map(fn ($name) => [
            'route_name' => $name,
            'message' => 'Sidebar/menu references route [' . $name . '] but it was not found in the registered route catalog.',
            'hint' => 'Check provider boot order, route namespaces, and route name drift.',
        ])->all();

        return [
            'status' => count($items) ? 'warn' : 'pass',
            'detail' => count($items) ? count($items) . ' unresolved named route(s) explained.' : 'No route-name mismatches detected.',
            'items' => $items,
        ];
    }
}
