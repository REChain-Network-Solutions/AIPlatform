<?php

namespace App\Support\ModuleDoctor;

class RegisteredRouteAuditService
{
    public function inspect(array $context, array $routeNames = []): array
    {
        $catalogRoutes = app(RegisteredRouteCatalogService::class)->all();
        $registered = collect($catalogRoutes);
        $missing = [];

        if ($registered->count()) {
            foreach ($routeNames as $routeName) {
                if (!$registered->contains($routeName)) {
                    $missing[] = $routeName;
                }
            }
        }

        return [
            'status' => empty($routeNames) ? 'warn' : (count($missing) ? 'warn' : 'pass'),
            'detail' => empty($routeNames)
                ? 'No named route references were found in the uploaded module.'
                : (count($missing)
                    ? count($missing) . ' named route(s) are referenced in menu/sidebar files but are not present in the known route catalog.'
                    : 'All discovered menu/sidebar route names were resolved.'),
            'catalog' => $catalogRoutes,
            'missing_route_names' => array_values(array_unique($missing)),
            'resolved_route_names' => array_values(array_unique(array_diff($routeNames, $missing))),
        ];
    }
}
