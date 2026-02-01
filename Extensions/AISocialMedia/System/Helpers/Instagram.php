<?php

namespace App\Extensions\AISocialMedia\System\Helpers;

use Exception;
use Illuminate\Foundation\Application;
use Illuminate\Http\Client\PendingRequest;
use Illuminate\Http\RedirectResponse;
use Illuminate\Routing\Redirector;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Log;

class Instagram
{
    protected array $config = [];

    public function __construct(?array $config = null, protected ?string $accessToken = null)
    {
        $instagramConfig = config('ai-social-media.instagram');

        $instagramConfig['app_id'] = setting('instagram_app_id');
        $instagramConfig['app_secret'] = setting('instagram_app_secret');

        $this->config = $config ?? $instagramConfig;
        $this->config['redirect_uri'] = secure_url(config('ai-social-media.instagram.redirect_uri'));
    }

    private function apiUrl(string $endpoint, array $params = [], bool $isBaseUrl = false)
    {
        $apiUrl = $isBaseUrl ? $this->config['base_url'] : $this->config['api_url'];

        if (str_starts_with($endpoint, '/')) {
            $endpoint = substr($endpoint, 1);
        }

        $v = $this->config['api_version'];
        $versionedUrlWithEndpoint = $apiUrl . '/' . ($v ? ($v . '/') : '') . $endpoint;

        if (count($params)) {
            $versionedUrlWithEndpoint .= '?' . http_build_query($params);
        }

        return $versionedUrlWithEndpoint;
    }

    private function apiClient(): PendingRequest
    {
        return Http::withToken($this->accessToken)
            ->baseUrl($this->config['api_url'])
            ->retry(1, 3000);
    }

    public function setToken(string $bearerToken)
    {
        $this->accessToken = $bearerToken;

        return $this;
    }

    public static function authRedirect(array $scopes = []): Application|Redirector|RedirectResponse|\Illuminate\Contracts\Foundation\Application
    {
        $instagram = new self;
        if ($scopes) {
            $instagram->config['scopes'] = $scopes;
        }
        $authUri = $instagram->apiUrl('dialog/oauth', [
            'response_type' => 'code',
            'client_id'     => $instagram->config['app_id'],
            'redirect_uri'  => $instagram->config['redirect_uri'],
            'scope'         => collect($instagram->config['scopes'])->join(','),
        ], true);

        return redirect($authUri);
    }

    public function getAccessToken(string $code): \Illuminate\Http\Client\Response
    {
        $redirect_uri = $this->apiUrl('/oauth/access_token', [
            'code'          => $code,
            'client_id'     => $this->config['app_id'],
            'client_secret' => $this->config['app_secret'],
            'redirect_uri'  => $this->config['redirect_uri'],
        ]);

        return Http::post($redirect_uri);
    }

    public function refreshAccessToken(): \Illuminate\Http\Client\Response
    {
        $apiUrl = $this->apiUrl('/oauth/access_token', [
            'client_id'         => $this->config['app_id'],
            'client_secret'     => $this->config['app_secret'],
            'grant_type'        => 'fb_exchange_token',
            'fb_exchange_token' => $this->accessToken,
        ]);

        return Http::post($apiUrl);
    }

    public function getAccountInfo(?array $fields = null): \Illuminate\Http\Client\Response
    {
        $redirect_uri = $this->apiUrl('/me/accounts', [
            'access_token' => $this->accessToken,
            'fields'       => collect($fields)->join(','),
        ]);

        return Http::get($redirect_uri);
    }

    public function getInstagramInfo(string $igId, ?array $fields = null): \Illuminate\Http\Client\Response
    {
        $redirect_uri = $this->apiUrl('/' . $igId);

        return Http::withToken($this->accessToken)->get($redirect_uri, [
            'fields' => collect($fields)->join(','),
        ]);
    }

    public function publishSingleMediaPost(string $igId, array $postData): \Illuminate\Http\Client\Response
    {
        $apiUrl = $this->apiUrl("$igId/media");

        $uploadMediaRes = Http::withToken($this->accessToken)
            ->retry(3, 3000)
            ->post($apiUrl, $postData)->throw();

        $mediaId = $uploadMediaRes->json('id');

        $uploadStatus = $this->checkUploadStatus($mediaId);

        throw_if(! $uploadStatus['is_ready'], new Exception($uploadStatus['status']));

        return $this->publishContainer($igId, $uploadMediaRes->json('id'));
    }

    public function publishCarouselPost(string $igId, array $files, string $mediaType = 'image', string $caption = ''): \Illuminate\Http\Client\Response
    {
        $containerIds = [];
        foreach ($files as $fileUrl) {
            $containerData = [
                'is_carousel_item' => true,
            ];

            if ($mediaType == 'image') {
                $containerData['media_type'] = 'IMAGE';
                $containerData['image_url'] = $fileUrl;
            } elseif ($mediaType == 'video') {
                $containerData['media_type'] = 'VIDEO';
                $containerData['video_url'] = $fileUrl;
            }

            $apiUrl = $this->apiUrl($igId . '/media');
            $containerRes = Http::withToken($this->accessToken)
                ->asForm()
                ->acceptJson()
                ->post($apiUrl, $containerData)
                ->throw();

            $containerIds[] = $containerRes->json('id');
        }

        $publishCarouselContainerRes = Http::withToken($this->accessToken)
            ->retry(3, 3000)
            ->post($apiUrl, [
                'media_type' => 'CAROUSEL',
                'children'   => $containerIds,
            ]);

        return $this->publishContainer($igId, $publishCarouselContainerRes->json('id'));
    }

    protected function publishContainer(string $igId, string $creation_id)
    {
        $apiUrl = $this->apiUrl($igId . '/media_publish');

        return Http::retry(3, 3000)
            ->withToken($this->accessToken)
            ->post($apiUrl, [
                'creation_id' => (int) $creation_id,
            ]);
    }

    private function checkUploadStatus(string $mediaId, int $delayInSeconds = 3, int $maxAttempts = 10): array
    {
        $status = false;
        $attempted = 0;
        $isFinished = false;

        while (! $isFinished && $attempted < $maxAttempts) {
            Log::info("Checking upload for: $mediaId");
            $videoStatus = $this->apiClient()->get($this->apiUrl($mediaId, ['fields' => 'status_code,status']))->throw();
            Log::info("Got upload status is: $status. on $attempted/$maxAttempts attempts");

            $status = $videoStatus->json('status_code');
            $isFinished = in_array(strtolower($status), ['finished', 'ok', 'completed', 'ready']);

            if ($isFinished) {
                Log::info("Upload finished with status: $status");

                break;
            }

            $isError = in_array(strtolower($status), ['error', 'failed']);
            if ($isError) {
                Log::info("Upload error with status: $status");

                break;
            }

            $attempted++;
            sleep($delayInSeconds);
        }

        return [
            'is_ready'    => $isFinished,
            'status_code' => $status,
            'status'      => $videoStatus->json('status'),
        ];
    }

    private function getMediaStatus(string $mediaId): \Illuminate\Http\Client\Response
    {
        $apiUrl = $this->apiUrl($mediaId, [
            'fields' => 'status',
        ]);

        return Http::withToken($this->accessToken)->get($apiUrl)->throw();
    }

    // analytics
    public function getPostAnalytics(string $postId, array $fields = []): \Illuminate\Http\Client\Response
    {
        return Http::withToken($this->accessToken)
            ->get($this->apiUrl($postId, [
                'fields' => collect($fields)->join(','),
            ]));
    }
}
