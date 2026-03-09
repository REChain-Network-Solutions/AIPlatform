<?php

namespace App\Domains\Marketplace\Services;

use App\Domains\Marketplace\Repositories\Contracts\ExtensionRepositoryInterface;
use App\Helpers\Classes\MarketplaceHelper;
use App\Helpers\Classes\VersionComparator;
use App\Models\Gateways;
use App\Services\Extension\ExtensionService;
use Database\Seeders\MenuSeeder;
use Exception;
use Illuminate\Support\Facades\Artisan;
use Illuminate\Support\Facades\Cache;
use Illuminate\Support\Facades\File;
use Illuminate\Support\Facades\Storage;
use ZipArchive;

class ExtensionInstallService
{
    public string $extensionInstallCache = 'new_extension_installed';

    public function __construct(
        public ZipArchive $archive,
        public ExtensionRepositoryInterface $repository
    ) {}

    public function install(string $slug): array
    {
        $extension = $this->repository->findBySlugInDb($slug);

        if ($slug === 'checkout-registration') {
            $gateway = Gateways::query()
                ->where('is_active', '0')
                ->where('code', 'stripe')
                ->first();

            if ($gateway) {
                return [
                    'status'  => false,
                    'message' => trans('This extension is not available for installation. Please activate the Stripe payment gateway first.'),
                ];
            }
        }

        $responseExtension = $this->repository->findSupport($extension->getAttribute('slug'));

        if (isset($responseExtension['parent']) && $parent = $responseExtension['parent']) {
            if (! MarketplaceHelper::isRegistered($parent['registration_key'])) {
                return [
                    'status'  => false,
                    'message' => 'To install this extension, you must first install the parent extension: ' . $parent['name'] . '.',
                ];
            }
        }

        $appVersion = $this->repository->appVersion();

        if (isset($responseExtension['min_app_version']) && VersionComparator::compareVersion($appVersion, $responseExtension['min_app_version'], '<')) {
            return [
                'status'  => false,
                'message' => trans('To download the extension, your app version must be at least ' . $responseExtension['min_app_version'] . '. Please update your system first.'),
            ];
        }

        $extensionFolderName = $responseExtension['extension_folder'];

        $extensionFolderPath = $this->mkdir($extensionFolderName);

        if (empty($extensionFolderPath)) {
            return [
                'status'  => false,
                'message' => 'Failed to create extension folder',
            ];
        }

        $response = $this->repository->install(
            $extension->getAttribute('slug'),
            data_get($responseExtension, 'version')
        );

        if ($response->failed()) {
            return [
                'status'  => false,
                'message' => trans('Failed to download extension'),
            ];
        }

        $zipContent = $response->body();

        $zipPath = $extensionFolderName . DIRECTORY_SEPARATOR . $slug . '.zip';

        Storage::disk('extension')->put($zipPath, $zipContent);

        $open = $this->archive->open(
            Storage::disk('extension')->path($zipPath)
        );

        $migrate = false;

        if ($open) {
            $this->archive->extractTo($extensionFolderPath);

            $this->archive->close();

            if (File::exists(resource_path('extensions/' . $slug . '/index.json'))) {
                try {
                    app(ExtensionService::class)->uninstall(
                        extensionSlug: $slug, newVersion: false, uninstall: false
                    );
                } catch (Exception $exception) {
                }
            }

            app(MenuSeeder::class)->run();

            Artisan::call('optimize:clear');

            Artisan::call('cache:clear');

            $migrate = Artisan::call('migrate', ['--force' => true]);

            Artisan::call('vendor:publish', [
                '--tag'   => 'extension',
                '--force' => true,
            ]);

            Storage::disk('extension')->delete($zipPath);

            $extension
                ->update([
                    'installed' => 1,
                    'version'   => data_get($responseExtension, 'version'),
                ]);

            Cache::remember($this->getExtensionInstallCache(), 60, function () {
                return true;
            });
        }

        return [
            'status'  => true,
            'data'    => $migrate,
            'message' => 'Extension installed',
        ];
    }

    public function mkdir($folder): string
    {
        if (Storage::disk('extension')->exists($folder)) {
            return Storage::disk('extension')->path($folder);
        }

        $mkCheck = Storage::disk('extension')->makeDirectory($folder);

        $folderPath = Storage::disk('extension')->path($folder);

        File::chmod($folderPath, 0777);

        return $mkCheck ? $folderPath : '';
    }

    public function getExtensionInstallCache(): string
    {
        return $this->extensionInstallCache;
    }
}
