<?php

namespace App\Extensions\AISocialMedia\System\Services;

use App\Extensions\AISocialMedia\System\Helpers\Instagram;
use App\Extensions\AISocialMedia\System\Services\Contracts\BaseService;
use Illuminate\Support\Facades\Storage;

class InstagramService extends BaseService
{
    public function share($text): void
    {
        $post = $this->getPost();

        $platform = $this->getPlatform();

        $postData = [
            'caption'   => $text,
            'image_url' => Storage::disk('public')->url('uploads/' . $post->media),
        ];

        if (! $setting = $platform->setting) {
            return;
        }

        $accessToken = $setting->getCredentialValue('access_token');

        $id = $setting->getCredentialValue('id');

        if (! $accessToken) {
            return;
        }

        if (! $id) {
            return;
        }

        // initialize the Instagram API class
        $instagram = new Instagram;
        // set the access token for the Instagram API class
        $instagram->setToken($accessToken);

        $instagram->publishSingleMediaPost($id, $postData);

        $post->update(['last_run_date' => now()]);
    }
}
