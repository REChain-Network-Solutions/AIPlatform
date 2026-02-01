<?php

namespace App\Extensions\AISocialMedia\System\Services;

use App\Extensions\AISocialMedia\System\Services\Contracts\BaseService;
use Exception;
use GuzzleHttp\Client;

class LinkedInService extends BaseService
{
    public string $accessToken;

    public function getProfile()
    {

        try {
            $client = new Client;

            $response = $client->request('GET', 'https://api.linkedin.com/v2/userinfo', [
                'headers' => [
                    'Authorization' => 'Bearer ' . $this->accessToken,
                ],
            ]);

            $object = json_decode($response->getBody()->getContents(), true);

            return $object;
        } catch (Exception $e) {
            return [];
        }

    }

    public function shareNone($text): void
    {
        $client = new Client;

        $profile = $this->getProfile();

        if (empty($profile)) {
            return;
        }

        if (! isset($profile['sub'])) {
            return;
        }

        $personURN = $profile['sub'];

        $client->request('POST', 'https://api.linkedin.com/v2/ugcPosts', [
            'headers' => [
                'Authorization'             => 'Bearer ' . $this->accessToken,
                'Connection'                => 'Keep-Alive',
                'Content-Type'              => 'application/json',
                'X-Restli-Protocol-Version' => '2.0.0',
            ],
            'json' => [
                'author'          => 'urn:li:person:' . $personURN,
                'lifecycleState'  => 'PUBLISHED',
                'specificContent' => [
                    'com.linkedin.ugc.ShareContent' => [
                        'shareCommentary' => [
                            'text' => $text,
                        ],
                        'shareMediaCategory' => 'NONE',
                    ],
                ],
                'visibility' => [
                    'com.linkedin.ugc.MemberNetworkVisibility' => 'PUBLIC',
                ],
            ],
        ]);

        $this->getPost()->update(['last_run_date' => now()]);
    }

    public function share($text): void
    {
        $this->accessToken = $this->getPlatform()->setting->getCredentialValue('access_token');

        if (! empty($text)) {
            $content = preg_replace('/\n+/', "\n", $text);

            $this->shareNone($content);
        }
    }
}
