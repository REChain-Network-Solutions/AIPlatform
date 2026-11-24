<?php

namespace App\Http\Controllers\Common;

use App\Helpers\Classes\Helper;
use App\Helpers\Classes\Localization;
use App\Http\Controllers\Controller;
use App\Http\Requests\UploadFilesRequest;
use App\Models\SettingTwo;
use App\Services\Security\RssFeedService;
use App\Services\Security\SafeUploadService;
use Elseyyid\LaravelJsonLocationsManager\Models\Strings;
use Illuminate\Contracts\Filesystem\FileNotFoundException;
use Illuminate\Filesystem\Filesystem;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\RedirectResponse;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Artisan;
use Illuminate\Support\Facades\Schema;
use Illuminate\Validation\ValidationException;
use JsonException;
use Mews\Purifier\Facades\Purifier;

class CommonController extends Controller
{
    public function translationsLangUpdate(Request $request, int $id): JsonResponse
    {
        $validated = $request->validate([
            'name'  => ['required', 'string', 'regex:/^[a-z]{2}(?:_[A-Z]{2})?$/'],
            'value' => ['required', 'string', 'max:20000'],
            'rich'  => ['sometimes', 'boolean'],
        ]);

        $this->assertTranslationColumn($validated['name']);

        Strings::query()->where('code', $id)->update([
            $validated['name'] => $this->sanitizeTranslation($validated['value'], $validated['rich'] ?? false),
        ]);

        return response()->json(['code' => 200]);
    }

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
        $validated = $request->validate([
            'setLocale' => ['required', 'string', 'regex:/^[a-z]{2}(?:_[A-Z]{2})?$/'],
        ]);
        $settings_two = SettingTwo::getCache();
        abort_unless(in_array($validated['setLocale'], explode(',', $settings_two->languages), true), 422);

        $settings_two->languages_default = $validated['setLocale'];
        $settings_two->save();
        Localization::setLocale($validated['setLocale']);

        return redirect()->route('elseyyid.translations.home', [$validated['setLocale']])->with(config('elseyyid-location.message_flash_variable'), $validated['setLocale']);
    }

    /**
     * @throws FileNotFoundException
     * @throws JsonException
     */
    public function translationsLangUpdateAll(Request $request): JsonResponse
    {
        $validated = $request->validate([
            'json' => ['required', 'json'],
            'lang' => ['required', 'string', 'regex:/^(?:edit|[a-z]{2}(?:_[A-Z]{2})?)$/'],
        ]);

        try {
            $json = json_decode($validated['json'], true, 512, JSON_THROW_ON_ERROR);
        } catch (JsonException) {
            throw ValidationException::withMessages(['json' => __('The translation payload is invalid.')]);
        }

        if (! is_array($json)) {
            throw ValidationException::withMessages(['json' => __('The translation payload is invalid.')]);
        }

        $column_name = $validated['lang'];
        $this->assertTranslationColumn($column_name);

        $updates = [];
        foreach ($json as $code => $column_value) {
            if (is_string($column_value) && $column_value !== '') {
                $updates[(string) $code] = $this->sanitizeTranslation($column_value);
            }
        }

        // Fail before touching the database when a value cannot be JSON-encoded,
        // so a chunk never commits rows it cannot persist to the language file.
        foreach ($updates as $column_value) {
            if (! is_string(json_encode($column_value))) {
                throw ValidationException::withMessages(['json' => __('The translation file could not be generated.')]);
            }
        }

        Strings::query()->getConnection()->transaction(static function () use ($updates, $column_name): void {
            foreach ($updates as $code => $column_value) {
                Strings::where('code', '=', $code)->update([
                    $column_name => $column_value,
                ]);
            }
        });

        // Filesystem failures after this point (disk full, permissions) are
        // operational: the database state is valid, retry the save to rewrite
        // the language files.
        $this->writeLanguageJson($column_name);

        if ($column_name === 'edit') {
            $this->mergeEditJsonIntoEn();
        }

        return response()->json(['code' => 200, 'saved' => count($updates)], 200);
    }

    private function writeLanguageJson(string $column): void
    {
        $list = Strings::pluck($column, 'en');
        $new_json = json_encode_prettify($list);

        if (! is_string($new_json)) {
            throw ValidationException::withMessages(['json' => __('The translation file could not be generated.')]);
        }

        (new Filesystem)->put(base_path('lang/' . $column . '.json'), $new_json);
    }

    private function mergeEditJsonIntoEn(): void
    {
        $filesystem = new Filesystem;

        try {
            // Read existing values from en.json
            $existingValues = json_decode($filesystem->get(base_path('lang/en.json')), true, 512, JSON_THROW_ON_ERROR);
            // Read non-empty values from edit.json
            $editValues = json_decode($filesystem->get(base_path('lang/edit.json')), true, 512, JSON_THROW_ON_ERROR);
        } catch (FileNotFoundException|JsonException) {
            throw ValidationException::withMessages(['lang' => __('The translation file could not be generated.')]);
        }

        if (! is_array($existingValues) || ! is_array($editValues)) {
            throw ValidationException::withMessages(['lang' => __('The translation file could not be generated.')]);
        }

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

        if (! is_string($updatedJson)) {
            throw ValidationException::withMessages(['lang' => __('The translation file could not be generated.')]);
        }

        // Write the updated JSON to en.json
        $filesystem->put(base_path('lang/en.json'), $updatedJson);
    }

    public function translationsLangSave(Request $request)
    {
        if (Helper::appIsDemo()) {
            return response()->json(['code' => 403, 'message' => __('This feature is disabled in demo mode.')], 403);
        }

        $validated = $request->validate([
            'lang'  => ['required', 'string', 'regex:/^[a-z]{2}(?:_[A-Z]{2})?$/'],
            'state' => ['required', 'boolean'],
        ]);

        $settings_two = SettingTwo::getCache();
        $codes = explode(',', $settings_two->languages);

        if ($validated['state']) {
            if (! in_array($validated['lang'], $codes)) {
                $codes[] = $validated['lang'];
            }
        } else {
            if (in_array($validated['lang'], $codes)) {
                unset($codes[array_search($validated['lang'], $codes)]);
            }
        }
        $settings_two->languages = implode(',', $codes);
        $settings_two->save();

        return response()->json(['code' => 200], 200);
    }

    private function assertTranslationColumn(string $column): void
    {
        $connection = (new Strings)->getConnectionName() ?? config('database.default');

        abort_unless(Schema::connection($connection)->hasColumn('strings', $column), 422, __('Unknown translation language.'));
    }

    private function sanitizeTranslation(string $value, bool $allowRichText = false): string
    {
        if (! $allowRichText) {
            return Purifier::clean($value, ['HTML.Allowed' => '']);
        }

        return Purifier::clean($value, [
            'HTML.Allowed'             => 'a[href|title|target],b,br,em,i,span[class],strong,u',
            'Attr.AllowedFrameTargets' => ['_blank'],
        ]);
    }

    public function imagesUpload(Request $request, SafeUploadService $safeUploadService): JsonResponse
    {
        $validated = $request->validate([
            'images'   => ['required', 'array', 'min:1', 'max:5'],
            'images.*' => ['required', 'string', 'max:30000000'],
        ]);
        $paths = [];
        $uploads = [];

        foreach ($validated['images'] as $image) {
            if (! preg_match('#^data:image/(jpeg|png|gif|webp);base64,#i', $image, $matches)) {
                abort(422, __('Only JPEG, PNG, GIF, and WebP images are allowed.'));
            }

            $extension = strtolower($matches[1]) === 'jpeg' ? 'jpg' : strtolower($matches[1]);
            $upload = $safeUploadService->store($request->user(), [
                'name' => 'image.' . $extension,
                'data' => $image,
            ]);
            $paths[] = route('uploads.show', $upload);
            $uploads[] = $upload->only(['id', 'original_name', 'mime_type', 'size']);
        }

        return response()->json(['path' => $paths, 'uploads' => $uploads]);
    }

    public function filesUpload(UploadFilesRequest $request, SafeUploadService $safeUploadService): JsonResponse
    {
        $images = [];
        $others = [];
        $uploads = [];

        foreach ($request->validated('files') as $file) {
            $upload = $safeUploadService->store($request->user(), $file);
            $url = route('uploads.show', $upload);
            $metadata = [
                'id'        => $upload->id,
                'name'      => $upload->original_name,
                'mime_type' => $upload->mime_type,
                'size'      => $upload->size,
                'url'       => $url,
            ];

            $uploads[] = $metadata;

            if (str_starts_with($upload->mime_type, 'image/')) {
                $images[] = $url;
            } else {
                $others[] = $url;
            }
        }

        // Priority: other > image
        if (! empty($others)) {
            return response()->json([
                'type'    => 'other',
                'path'    => $others,
                'uploads' => $uploads,
            ]);
        }

        return response()->json([
            'type'    => 'image',
            'path'    => $images,
            'uploads' => $uploads,
        ]);
    }

    public function imageUpload(Request $request, SafeUploadService $safeUploadService): JsonResponse
    {
        $validated = $request->validate([
            'image' => ['required', 'file', 'mimes:jpg,jpeg,png,gif,webp', 'max:20480'],
        ]);
        $image = $validated['image'];
        $contents = file_get_contents($image->getRealPath());
        abort_if($contents === false, 422, __('The image could not be read.'));

        $upload = $safeUploadService->store($request->user(), [
            'name' => $image->getClientOriginalName(),
            'data' => sprintf('data:%s;base64,%s', $image->getMimeType(), base64_encode($contents)),
        ]);

        return response()->json([
            'path'   => route('uploads.show', $upload),
            'upload' => $upload->only(['id', 'original_name', 'mime_type', 'size']),
        ]);
    }

    public function rssFetch(Request $request, RssFeedService $rssFeedService): JsonResponse
    {
        $validated = $request->validate(['url' => ['required', 'url:http,https', 'max:2048']]);
        $data = $rssFeedService->fetch($validated['url']);

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
