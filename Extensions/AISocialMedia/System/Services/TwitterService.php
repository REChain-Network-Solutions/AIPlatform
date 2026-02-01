<?php

namespace App\Extensions\AISocialMedia\System\Services;

use App\Extensions\AISocialMedia\System\Services\Contracts\BaseService;
use Illuminate\Support\Arr;
use Illuminate\Support\Fluent;
use Noweh\TwitterApi\Client;

class TwitterService extends BaseService
{
    public function share($text): void
    {
        $post = $this->getPost();

        $setting = $this->getPlatform()->setting;

        if (! $setting) {
            return;
        }

        $credentials = new Fluent($setting['credentials'] ?: []);

        $credentials = Arr::only($credentials->toArray(), [
            'account_id',
            'access_token',
            'access_token_secret',
            'consumer_key',
            'consumer_secret',
            'bearer_token',
        ]);

        $client = new Client($credentials);
        $client->tweet()->create()
            ->performRequest([
                'text' => str_replace('"', '', $text),
            ],
                withHeaders: true
            );

        $post->update(['last_run_date' => now()]);

    }
}
