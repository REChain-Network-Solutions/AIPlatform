<?php

namespace App\Services\Common;

use Exception;
use GuzzleHttp\Client;
use Illuminate\Support\Facades\Cache;
use Illuminate\Support\Facades\Storage;
use RuntimeException;

class FontsService
{
    public const FONTS_KEY = 'dynamic_fonts_key';

    public static function getGoogleFonts(): array
    {
        $cacheKey = self::FONTS_KEY;
        $fileName = 'google_fonts_cache.json';

        $fonts = Cache::rememberForever($cacheKey, static function () use ($fileName) {
            try {
                $client = new Client;
                $sort = 'popularity';

                $response = $client->get("https://www.googleapis.com/webfonts/v1/webfonts?key=AIzaSyBfkS4mQ0jGvdP13yAlEb88LHZ3ddRoCXU&sort={$sort}");
                if ($response->getStatusCode() !== 200) {
                    throw new RuntimeException('Google Fonts API returned status code: ' . $response->getStatusCode());
                }
                $body = $response->getBody()->getContents();
                if (empty($body)) {
                    throw new RuntimeException('Empty response from Google Fonts API');
                }
                $fonts = json_decode($body, true, 512, JSON_THROW_ON_ERROR);
                if (! isset($fonts['items']) || ! is_array($fonts['items'])) {
                    throw new RuntimeException('Unexpected or malformed response from Google Fonts API');
                }
                Storage::disk('local')->put($fileName, json_encode($fonts, JSON_THROW_ON_ERROR|JSON_UNESCAPED_UNICODE));

                return $fonts;
            } catch (RuntimeException|Exception $e) {
                if (Storage::disk('local')->exists($fileName)) {
                    $cachedFonts = json_decode(Storage::disk('local')->get($fileName), true, 512, JSON_THROW_ON_ERROR);
                    if (isset($cachedFonts['items']) && is_array($cachedFonts['items'])) {
                        return $cachedFonts;
                    }
                }

                return [];
            }
        });

        return $fonts ?? [];
    }

    public static function updateFontsCache(): array
    {
        Cache::forget(self::FONTS_KEY);

        return self::getGoogleFonts();
    }
}
