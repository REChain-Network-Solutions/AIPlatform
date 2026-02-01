<?php

namespace App\Extensions\AISocialMedia\System\Helpers;

use App\Models\Setting;

class AutomationHelper
{
    public static function apiKeys()
    {
        $settings = Setting::getCache();

        $apiKeys = explode(',', $settings->openai_api_secret);
        $apiKey = $apiKeys[array_rand($apiKeys)];

        $len = strlen($apiKey);
        $len = max($len, 6);
        $parts[] = substr($apiKey, 0, $l[] = rand(1, $len - 5));
        $parts[] = substr($apiKey, $l[0], $l[] = rand(1, $len - $l[0] - 3));
        $parts[] = substr($apiKey, array_sum($l));

        return [
            'apikeyPart1' => base64_encode($parts[0]),
            'apikeyPart2' => base64_encode($parts[1]),
            'apikeyPart3' => base64_encode($parts[2]),
            'apiUrl' 	    => base64_encode('https://api.openai.com/v1/chat/completions'),
        ];
    }
}
