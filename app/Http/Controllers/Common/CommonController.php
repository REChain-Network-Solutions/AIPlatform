<?php

namespace App\Http\Controllers\Common;

use App\Helpers\Classes\Helper;
use App\Helpers\Classes\Localization;
use App\Helpers\Classes\MarketplaceHelper;
use App\Http\Controllers\Controller;
use App\Models\SettingTwo;
use Exception;
use Illuminate\Contracts\Filesystem\FileNotFoundException;
use Illuminate\Http\File;
use Illuminate\Http\RedirectResponse;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Artisan;
use Illuminate\Support\Facades\Log;
use Illuminate\Support\Facades\Storage;
use Illuminate\Support\Str;
use JsonException;
use RuntimeException;
use Throwable;

class CommonController extends Controller
{
    public function regenerate()
    {
        Artisan::call('elseyyid:location:install');

        return redirect()->route('elseyyid.translations.home')->with(config('elseyyid-location.message_flash_variable'), __('Language files regenerated!'));
    }

    public function debug()
    {
        $currentDebugValue = env('APP_DEBUG', false);

        $newDebugValue = ! $currentDebugValue;

        $envContent = file_get_contents(base_path('.env'));

        $envContent = preg_replace('/^APP_DEBUG=.*/m', 'APP_DEBUG=' . ($newDebugValue ? 'true' : 'false'), $envContent);

        file_put_contents(base_path('.env'), $envContent);

        Artisan::call('config:clear');

        return redirect()->back()->with('message', 'Debug mode updated successfully.');
    }

    public function setLocale(Request $request): RedirectResponse
    {
        $settings_two = \App\Models\SettingTwo::getCache();
        $settings_two->languages_default = $request->setLocale;
        $settings_two->save();
        Localization::setLocale($request->setLocale);

        return redirect()->route('elseyyid.translations.home', [$request->setLocale])->with(config('elseyyid-location.message_flash_variable'), $request->setLocale);
    }

    /**
     * @throws FileNotFoundException
     * @throws JsonException
     */
    public function translationsLangUpdateAll(Request $request)
    {
        ini_set('max_input_vars', -1);
        $json = json_decode($request->input('json'), true, 512, JSON_THROW_ON_ERROR);
        $column_name = $request->get('lang');
        foreach ($json as $code => $column_value) {
            if (! empty($column_value)) {
                $test = \Elseyyid\LaravelJsonLocationsManager\Models\Strings::where('code', '=', $code)->update([$column_name => $column_value]);
            }
        }

        $lang = $column_name;
        $list = \Elseyyid\LaravelJsonLocationsManager\Models\Strings::pluck($lang, 'en');

        $new_json = json_encode_prettify($list);
        $filesystem = new \Illuminate\Filesystem\Filesystem;
        $filesystem->put(base_path('lang/' . $lang . '.json'), $new_json);

        if ($column_name == 'edit') {
            // Read existing values from en.json
            $enJsonPath = base_path('lang/en.json');
            $existingJson = $filesystem->get($enJsonPath);
            $existingValues = json_decode($existingJson, true);
            // Read non-empty values from edit.json
            $editJsonPath = base_path('lang/edit.json');
            $editJson = $filesystem->get($editJsonPath);
            $editValues = json_decode($editJson, true);
            // Update values in en.json using keys from edit.json
            foreach ($editValues as $key => $column_value) {
                // Check if the value is not empty
                if (! empty($column_value)) {
                    // Update the existing values with non-empty values using the key from edit.json
                    $existingValues[$key] = $column_value;
                }
            }
            // Convert the updated values to JSON
            $updatedJson = json_encode_prettify($existingValues);
            // Write the updated JSON to en.json
            $filesystem->put($enJsonPath, $updatedJson);
        }

        return response()->json(['code' => 200], 200);
    }

    public function translationsLangSave(Request $request)
    {
        if (Helper::appIsDemo()) {
            return response()->json(['code' => 403, 'message' => __('This feature is disabled in demo mode.')], 403);
        }

        $settings_two = \App\Models\SettingTwo::getCache();
        $codes = explode(',', $settings_two->languages);

        if ($request->state) {
            if (! in_array($request->lang, $codes)) {
                $codes[] = $request->lang;
            }
        } else {
            if (in_array($request->lang, $codes)) {
                unset($codes[array_search($request->lang, $codes)]);
            }
        }
        $settings_two->languages = implode(',', $codes);
        $settings_two->save();

        return response()->json(['code' => 200], 200);
    }

    public function imagesUpload(Request $request)
    {
        $images = $request->input('images');

        $paths = [];

        foreach ($images ?? [] as $image) {
            $base64Image = $image;
            $nameOfImage = Str::random(12) . '.png';

            // save file on local storage or aws s3
            Storage::disk('public')->put($nameOfImage, base64_decode(preg_replace('#^data:image/\w+;base64,#i', '', $base64Image)));
            $path = '/uploads/' . $nameOfImage;

            $uploadedFile = new File(substr($path, 1));

            if (SettingTwo::getCache()->ai_image_storage == 's3') {
                try {
                    $aws_path = Storage::disk('s3')->put('', $uploadedFile);
                    unlink(substr($path, 1));
                    $path = Storage::disk('s3')->url($aws_path);
                } catch (Exception $e) {
                    return response()->json(['status' => 'error', 'message' => 'AWS Error - ' . $e->getMessage()]);
                }
            }

            if (SettingTwo::getCache()->ai_image_storage == 'r2') {
                try {
                    $aws_path = Storage::disk('r2')->put('', $uploadedFile);
                    unlink(substr($path, 1));
                    $path = Storage::disk('r2')->url($aws_path);
                } catch (Exception $e) {
                    return response()->json(['status' => 'error', 'message' => 'AWS Error - ' . $e->getMessage()]);
                }
            }

            $paths[] = $path;
        }

        return response()->json(['path' => $paths]);
    }

    public function filesUpload(Request $request): \Illuminate\Http\JsonResponse
    {
        $contentManagerActive = MarketplaceHelper::isRegistered('content-manager')
            && setting('content_manager_enabled', '1');

        $images = [];
        $others = [];

        foreach ($request->input('files', []) as $file) {
            $name = $file['name'] ?? 'file';
            $base64Data = $file['data'] ?? null;

            if (! $base64Data) {
                continue;
            }

            if (preg_match('/^data:([^;]+);base64,(.+)$/', $base64Data, $matches)) {
                $mimeType = $matches[1];
                $rawData = base64_decode($matches[2]);
                $extension = mimeToExtension($mimeType);

                if (! pathinfo($name, PATHINFO_EXTENSION)) {
                    $name .= '.' . $extension;
                }
            } else {
                continue;
            }

            // Decide type
            $fileType = str_starts_with($mimeType, 'image/') ? 'images' : 'other';

            // Directory per user
            $relativePath = "uploads/media/{$fileType}/u-" . auth()->id();
            $absoluteDir = public_path($relativePath);

            if (! is_dir($absoluteDir) && ! mkdir($absoluteDir, 0775, true) && ! is_dir($absoluteDir)) {
                throw new RuntimeException(sprintf('Directory "%s" was not created', $absoluteDir));
            }

            // Avoid duplicates only for "other" when content manager active
            if ($fileType === 'other' && $contentManagerActive) {
                $existingPath = $absoluteDir . '/' . $name;
                if (file_exists($existingPath)) {
                    $publicPath = '/' . $relativePath . '/' . $name;
                    $uploadedFile = new File($existingPath);
                    $this->checkOtherStorage($uploadedFile, $publicPath);

                    $others[] = $publicPath;

                    continue;
                }
            }

            // File naming
            $uniqueName = ($fileType === 'other' && $contentManagerActive)
                ? $name
                : Str::random(10) . '-' . time() . '-' . $name;

            $absolutePath = $absoluteDir . '/' . $uniqueName;

            try {
                if (file_put_contents($absolutePath, $rawData) === false) {
                    Log::error("Failed to save file: {$absolutePath}");

                    continue;
                }

                $publicPath = '/' . $relativePath . '/' . $uniqueName;
                $uploadedFile = new File($absolutePath);
                $this->checkOtherStorage($uploadedFile, $publicPath);

                if (! validateUploadedFile(public_path($publicPath), $extension)) {
                    Storage::disk('public')->delete($publicPath);

                    continue;
                }

                if ($fileType === 'images') {
                    $images[] = $publicPath;
                } else {
                    $others[] = $publicPath;
                }
            } catch (Throwable $e) {
                Log::error("Upload failed for {$uniqueName}: " . $e->getMessage());
            }
        }

        // Priority: other > image
        if (! empty($others)) {
            return response()->json([
                'type' => 'other',
                'path' => $others,
            ]);
        }

        return response()->json([
            'type' => 'image',
            'path' => $images,
        ]);
    }

    private function checkOtherStorage($uploadedFile, &$path): void
    {
        try {
            $disk = SettingTwo::getCache()->ai_image_storage;

            if (in_array($disk, ['s3', 'r2'])) {
                $awsPath = Storage::disk($disk)->putFile('', $uploadedFile);
                unlink($uploadedFile->getPathname());
                $path = Storage::disk($disk)->url($awsPath);
            }
        } catch (Exception $e) {
            Log::error($e->getMessage());
        }
    }

    public function imageUpload(Request $request)
    {
        $image = $request->file('image');
        $title = $request->input('title');

        $imageContent = file_get_contents($image->getRealPath());
        $base64Image = base64_encode($imageContent);
        $nameOfImage = Str::random(12) . '.png';

        // save file on local storage or aws s3
        Storage::disk('public')->put($nameOfImage, base64_decode($base64Image));
        $path = '/uploads/' . $nameOfImage;
        $uploadedFile = new File(substr($path, 1));

        if (SettingTwo::getCache()->ai_image_storage == 's3') {
            try {
                $aws_path = Storage::disk('s3')->put('', $uploadedFile);
                unlink(substr($path, 1));
                $path = Storage::disk('s3')->url($aws_path);
            } catch (Exception $e) {
                return response()->json(['status' => 'error', 'message' => 'AWS Error - ' . $e->getMessage()]);
            }
        }

        if (SettingTwo::getCache()->ai_image_storage == 'r2') {
            try {
                $aws_path = Storage::disk('r2')->put('', $uploadedFile);
                unlink(substr($path, 1));
                $path = Storage::disk('r2')->url($aws_path);
            } catch (Exception $e) {
                return response()->json(['status' => 'error', 'message' => 'AWS Error - ' . $e->getMessage()]);
            }
        }

        return response()->json(['path' => "$path"]);
    }

    public function rssFetch(Request $request)
    {
        $data = parseRSS($request->url);

        if (! $data) {
            return response()->json(__('RSS Not Fetched! Please check your URL and validete the RSS!'), 419);

        }

        $html = '';

        foreach ($data as $post) {
            $html .= sprintf(
                '<option value="%1$s" data-image="%2$s">%1$s</option>',
                e($post['title']),
                e($post['image']),
            );
        }

        return response()->json($html, 200);
    }
}
