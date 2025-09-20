<?php

namespace App\Services\Extension;

use App\Domains\Marketplace\Repositories\Contracts\ExtensionRepositoryInterface;
use App\Services\Extension\Traits\InstallExtension;
use App\Services\Extension\Traits\UninstallExtension;
use Illuminate\Support\Facades\File;
use ZipArchive;
use App\Models\Extension;
use Illuminate\Filesystem\Filesystem;
use Illuminate\Support\Facades\Storage;
use Illuminate\Support\Facades\Artisan;
use Exception;
use Illuminate\Support\Facades\Log;
use Illuminate\Support\Facades\View;

class ExtensionService
{
    use InstallExtension;
    use UninstallExtension;

    public string $zipExtractPath;

    public string $extensionSlug;

    public string $indexJson;

    public array $indexJsonArray;

    public function __construct(
        public ZipArchive $zipArchive,
        public ExtensionRepositoryInterface $extensionRepository
    ) {
    }

    public function deleteOldVersionFiles(): void
    {
        $data = data_get($this->indexJsonArray, 'delete_old_version_files');

        if (empty($data) && !is_array($data)) {
            return;
        }

        foreach ($data as $file) {
            $destinationPath = base_path($file);

            if (File::exists($destinationPath)) {
                File::delete($destinationPath);
            }
        }
    }

    public function makeDir(?string $extensionSlug = null): void
    {
        $extensionSlug = $extensionSlug ?? $this->extensionSlug;

        // make resource dir for extension
        if (!File::isDirectory(resource_path("extensions/$extensionSlug/"))) {
            File::makeDirectory(resource_path("extensions/$extensionSlug/"), 0777, true);
        }

        // make resource dir for extension
        if (!File::isDirectory(resource_path("extensions/$extensionSlug/migrations/uninstall"))) {
            File::makeDirectory(resource_path("extensions/$extensionSlug/migrations/uninstall"), 0777, true);
        }

        // make routes dir for extension
        if (!File::isDirectory(base_path('routes/extroutes/'))) {
            File::makeDirectory(base_path('routes/extroutes/'), 0777, true);
        }

        // make header views dir for extension
        if (!File::isDirectory(resource_path('views/default/components/navbar/extnavbars'))) {
            File::makeDirectory(resource_path('views/default/components/navbar/extnavbars'), 0777, true);
        }
    }

    /**
     * Get index.json from extracted zip
     */
    public function getIndexJson(?string $zipExtractPath = null): bool|string
    {
        $zipExtractPath = $zipExtractPath ?? $this->zipExtractPath;

        $path = $this->getZipJsonPath($zipExtractPath);

        if (!File::exists($path)) {
            return false;
        }

        $this->indexJson = file_get_contents(
            $this->getZipJsonPath($zipExtractPath)
        );

        if ($this->indexJson) {
            $this->indexJsonArray = json_decode($this->indexJson, true);
        }

        return $this->indexJson;
    }

    /**
     * Extracted zip json path
     */
    public function getZipJsonPath(?string $zipExtractPath = null): string
    {
        $zipExtractPath = $zipExtractPath ?? $this->zipExtractPath;

        return $zipExtractPath . DIRECTORY_SEPARATOR . 'index.json';
    }

    public function install(string $extensionSlug): bool|array
    {
        $dbExtension = Extension::query()->where('slug', $extensionSlug)->first();

        $this->extensionSlug = $extensionSlug;

        $responseExtension = $this->extensionRepository->find($dbExtension->getAttribute('slug'));

        $extensionFolderName = $responseExtension['extension_folder'];

        if ($extensionFolderName && $this->extensionRepository->appVersion() >= 7.3) {
            try {
                app(ExtensionInstallService::class)
                    ->install($extensionSlug);
                return ['success' => true, 'message' => 'Installed using ExtensionInstallService'];
            } catch (\Exception $e) {
                Log::error("Failed to install extension {$extensionSlug}: " . $e->getMessage());
                return ['success' => false, 'message' => $e->getMessage()];
            }
        }

        $version = data_get($responseExtension, 'version');

        $response = $this->extensionRepository->install(
            $dbExtension->getAttribute('slug'),
            $version
        );

        if ($response->failed()) {
            return [
                'success' => false,
                'message' => trans('Failed to download extension'),
            ];
        }

        $zipContent = $response->body();

        Storage::disk('local')->put('file.zip', $zipContent);

        $checkZip = $this->zipArchive->open(
            Storage::disk('local')->path('file.zip')
        );

        if ($checkZip) {

            $this->zipExtractPath = storage_path('app/zip-extract');

            $this->zipArchive->extractTo($this->zipExtractPath);

            $this->zipArchive->close();

            Storage::disk('local')->delete('file.zip');

            try {
                // index json
                $this->getIndexJson();

                if (empty($this->indexJsonArray)) {
                    return [
                        'success' => false,
                        'message' => trans('index.json not found'),
                    ];
                }

                $this->deleteOldVersionFiles();

                // make dir
                $this->makeDir($extensionSlug);

                // run install query
                $this->runInstallQuery();

                // copy resource
                $this->copyResource();

                // copy view
                $this->copyRoute();

                // copy controllers
                $this->copyControllers();

                // copy files
                $this->copyFiles();

                // if has migration files
                if (data_get($this->indexJsonArray, 'migration')) {
                    Artisan::call('migrate');
                }

                // delete zip extract dir
                (new Filesystem)->deleteDirectory($this->zipExtractPath);

                Extension::query()->where('slug', $extensionSlug)
                    ->update([
                        'installed' => 1,
                        'version' => data_get($this->indexJsonArray, 'version'),
                    ]);

                Artisan::call('cache:clear');

                return [
                    'success' => true,
                    'message' => trans('Extension installed successfully'),
                ];
            } catch (Exception $e) {
                return [
                    'success' => false,
                    'message' => $e->getMessage(),
                ];
            }
        }
    }
}