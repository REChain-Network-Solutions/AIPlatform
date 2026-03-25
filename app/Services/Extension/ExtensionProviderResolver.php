<?php

declare(strict_types=1);

namespace App\Services\Extension;

use App\Domains\Marketplace\MarketplaceServiceProvider;
use App\Models\Extension;
use Illuminate\Database\QueryException;
use Illuminate\Support\Facades\Log;
use Illuminate\Support\Facades\Schema;
use Illuminate\Support\Str;
use Illuminate\Support\Facades\File;

class ExtensionProviderResolver
{
    public function resolve(): array
    {
        $providers = [];

        // Always keep legacy fallback (Chatbot) to avoid regressions when DB is not ready.
        $providers[] = \App\Providers\ChatbotServiceProvider::class;

        if (! $this->canQueryExtensions()) {
            return array_values(array_unique(array_filter($providers, 'class_exists')));
        }

        $installedExtensions = Extension::query()
            ->where('installed', true)
            ->get(['slug']);

        foreach ($installedExtensions as $extension) {
            $slug = $extension->slug;

            if ($this->isNestedExtensionsPath($slug)) {
                Log::warning('Skipping extension with nested path', ['slug' => $slug]);

                continue;
            }

            $provider = $this->resolveProviderForSlug($slug);

            if ($provider && class_exists($provider)) {
                $providers[] = $provider;
            } else {
                Log::warning('Extension provider not found', [
                    'slug'      => $slug,
                    'candidate' => $provider,
                ]);
            }
        }

        return array_values(array_unique($providers));
    }

    private function resolveProviderForSlug(string $slug): ?string
    {
        // 1) Marketplace static map (keeps compatibility)
        $map = MarketplaceServiceProvider::getExtensionProviders();
        if (array_key_exists($slug, $map) && class_exists($map[$slug])) {
            return $map[$slug];
        }

        // 2) Provider declared in stored index.json (resources/extensions/{slug}/index.json)
        $manifestProvider = $this->providerFromIndexJson($slug);
        if ($manifestProvider && class_exists($manifestProvider)) {
            return $manifestProvider;
        }

        // 3) Conventional path \App\Extensions\<Studly>\System\<Studly>ServiceProvider
        $studly = Str::studly(str_replace(['-', '_'], ' ', $slug));
        $convention = "\\App\\Extensions\\{$studly}\\System\\{$studly}ServiceProvider";

        return $convention;
    }

    private function providerFromIndexJson(string $slug): ?string
    {
        $manifestPath = resource_path("extensions/{$slug}/index.json");

        if (! File::exists($manifestPath)) {
            return null;
        }

        $contents = json_decode(File::get($manifestPath), true);

        if (! is_array($contents)) {
            return null;
        }

        $provider = data_get($contents, 'provider') ?? data_get($contents, 'provider_class');

        return is_string($provider) && $provider !== '' ? $provider : null;
    }

    private function isNestedExtensionsPath(string $slug): bool
    {
        return Str::contains($slug, ['Extensions/Extensions', 'extensions/extensions']);
    }

    private function canQueryExtensions(): bool
    {
        try {
            return Schema::hasTable('extensions');
        } catch (QueryException $e) {
            Log::warning('ExtensionProviderResolver unable to query extensions table', ['error' => $e->getMessage()]);

            return false;
        }
    }
}
