<?php

namespace Modules\Inspection\Services;

use Illuminate\Support\Facades\Route;

class TitanLinkService
{
    public function docsUrl(): string
    {
        $route = config('inspection_integrations.titan_docs.route');
        if ($route && Route::has($route)) {
            return route($route);
        }
        return url(config('inspection_integrations.titan_docs.url', '/account/titan-docs'));
    }

    public function complianceUrl(): string
    {
        $route = config('inspection_integrations.titan_compliance.route');
        if ($route && Route::has($route)) {
            return route($route);
        }
        return url(config('inspection_integrations.titan_compliance.url', '/account/titan-compliance'));
    }

    public function docsLabel(): string
    {
        return __('inspection::general.titan_docs_label', [], null)
            ?: (string) config('inspection_integrations.titan_docs.label', 'Open in Titan Docs');
    }

    public function complianceLabel(): string
    {
        return __('inspection::general.titan_compliance_label', [], null)
            ?: (string) config('inspection_integrations.titan_compliance.label', 'Open in Titan Compliance');
    }
}
