<?php

namespace App\Extensions\AiAvatar\System\Services\Traits;

use Illuminate\Support\Facades\Storage;
use Psr\Http\Message\ResponseInterface;

trait AiAvatar
{
    public function createVideo(
        array $input,
        string $visibility,
        string $title,
        string $description,
        string $test
    ): array {
        $body = [
            'input'       => $input,
            'test'        => $test,
            'visibility'  => $visibility,
            'title'       => $title,
            'description' => $description,
        ];

        return $this->post('/videos', $body);
    }

    public function listVideos(int $limit = 20, int $offset = 0): array
    {
        $url = "/videos?limit={$limit}&offset={$offset}";

        return $this->get($url);
    }

    public function retrieveVideo(string $videoId): array
    {
        $url = "/videos/{$videoId}";

        return $this->get($url);
    }

    public function deleteVideo(string $videoId): ResponseInterface
    {
        $url = "/videos/{$videoId}";

        return $this->delete($url);
    }

    public static function listAvatars(): array
    {
        $file = Storage::disk('data')->get('synthesia/avatars.json');

        return json_decode($file, true, 512, JSON_THROW_ON_ERROR);
    }

    public static function listBackgrounds(): array
    {
        $file = Storage::disk('data')->get('synthesia/backgrounds.json');

        return json_decode($file, true);
    }
}
