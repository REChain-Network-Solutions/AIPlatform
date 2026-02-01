<?php

namespace App\Extensions\Wordpress\System\Services;

use App\Models\Integration\UserIntegration;
use Exception;
use GuzzleHttp\Client;
use GuzzleHttp\Exception\GuzzleException;
use JetBrains\PhpStorm\NoReturn;

class Wordpress
{
    public const LOGIN_URL = '/wp-json/api/v1/token';

    public const POST_URL = '/wp-json/wp/v2/posts';

    public const CATEGORY_URL = '/wp-json/wp/v2/categories';

    public const TAG_URL = '/wp-json/wp/v2/tags';

    public const IMAGE_URL = '/wp-json/wp/v2/media';

    protected ?string $domain;

    protected ?string $username;

    protected ?string $password;

    protected Client $client;

    protected null|bool|string $token;

    public function __construct(UserIntegration $userIntegration)
    {
        $credentials = $userIntegration->credentials;

        if (is_array($credentials)) {
            $this->domain = self::formatToHttps(data_get($credentials, 'domain.value'));
            $this->username = data_get($credentials, 'username.value');
            $this->password = data_get($credentials, 'password.value');

            $this->setToken();
        }

    }

    public function setToken(): void
    {
        $this->client = new Client;

        $this->token = $this->login();
    }

    public function addImage(array $data = [])
    {
        if (is_null($this->token)) {
            throw new Exception('Token is not set');
        }

        try {

            $response = $this->client->post($this->domain . self::IMAGE_URL, [
                'headers' => [
                    'Authorization' => 'Bearer ' . $this->token,
                    'timeout'       => 50,
                ],
                'multipart' => [
                    [
                        'name'     => 'title',
                        'contents' => data_get($data, 'title'),
                    ],
                    [
                        'name'     => 'file',
                        'contents' => data_get($data, 'file'),
                    ],
                ],
            ]);

            $statusCode = $response->getStatusCode();
            $body = json_decode($response->getBody(), true);

            if ($statusCode == 200 || $statusCode == 201) {
                return true;
            } else {
                throw new Exception('Error while creating post: ' . json_encode($body));
            }
        } catch (\GuzzleHttp\Exception\GuzzleException $e) {
            throw new Exception('Error while creating post: ' . $e->getMessage());
        }
    }

    public function create(array $data = [])
    {
        if (is_null($this->token)) {
            throw new Exception('Token is not set');
        }

        try {
            $response = $this->client->post($this->domain . self::POST_URL, [
                'headers' => [
                    'Authorization' => 'Bearer ' . $this->token,
                    'timeout'       => 50,
                ],
                'form_params' => [
                    'title'          => data_get($data, 'title'),
                    'content'        => data_get($data, 'content'),
                    'status'         => data_get($data, 'status'),
                    'comment_status' => data_get($data, 'comment_status'),
                    'categories'     => data_get($data, 'categories'),
                    'tags'           => data_get($data, 'tags'),
                    'featured_media' => data_get($data, 'featured_media'),
                    'date_gmt'       => data_get($data, 'date_gmt'),
                ],
            ]);

            if ($response->getStatusCode() == 200) {
                return json_decode($response->getBody(), true);
            }

        } catch (\GuzzleHttp\Exception\GuzzleException $e) {
            throw new Exception('Error while creating post');
        }
    }

    #[NoReturn]
    public function login(): null|bool|string
    {
        try {
            $response = $this->client->post($this->domain . self::LOGIN_URL, [
                'form_params' => [
                    'username' => $this->username,
                    'password' => $this->password,
                ],
            ]);

            $body = json_decode($response->getBody(), true);

            if ($response->getStatusCode() == 200 && isset($body['jwt_token'])) {
                $this->token = $body['jwt_token'];

                return $this->token;
            }

            return false;
        } catch (GuzzleException $e) {
            throw new Exception('Invalid credentials');
        }
    }

    public static function form(array $data = []): array
    {
        return [
            'domain' => [
                'type'  => 'text',
                'name'  => 'domain',
                'label' => 'Domain',
                'value' => old('credentials.domain', data_get($data, 'domain')),
            ],
            'username' => [
                'type'  => 'text',
                'name'  => 'username',
                'label' => 'Username',
                'value' => old('credentials.username', data_get($data, 'username')),
            ],
            'password' => [
                'type'  => 'password',
                'name'  => 'password',
                'label' => 'Password',
                'value' => old('credentials.password', data_get($data, 'password')),
            ],
        ];
    }

    public function category()
    {
        if (is_null($this->token)) {
            throw new Exception('Token is not set');
        }

        try {
            $response = $this->client->get($this->domain . self::CATEGORY_URL, [
                'headers' => [
                    'Authorization' => 'Bearer ' . $this->token,
                    'timeout'       => 50,
                ],
            ]);

            if ($response->getStatusCode() == 200) {
                return json_decode($response->getBody(), true);
            }

        } catch (\GuzzleHttp\Exception\GuzzleException $e) {
            throw new Exception('Error while creating post');
        }
    }

    public function tags()
    {
        if (is_null($this->token)) {
            throw new Exception('Token is not set');
        }

        try {
            $response = $this->client->get($this->domain . self::TAG_URL, [
                'headers' => [
                    'Authorization' => 'Bearer ' . $this->token,
                    'timeout'       => 50,
                ],
            ]);

            if ($response->getStatusCode() == 200) {
                return json_decode($response->getBody(), true);
            }

        } catch (\GuzzleHttp\Exception\GuzzleException $e) {
            throw new Exception('Error while creating post');
        }
    }

    public function images()
    {
        if (is_null($this->token)) {
            throw new Exception('Token is not set');
        }

        try {
            $response = $this->client->get($this->domain . self::IMAGE_URL, [
                'headers' => [
                    'Authorization' => 'Bearer ' . $this->token,
                    'timeout'       => 50,
                ],
            ]);

            if ($response->getStatusCode() == 200) {
                return json_decode($response->getBody(), true);
            }

        } catch (\GuzzleHttp\Exception\GuzzleException $e) {
            throw new Exception('Error while creating post');
        }
    }

    private static function formatToHttps($url)
    {
        return 'https://' . preg_replace('/^(http:\/\/|https:\/\/|www\.)/', '', $url);
    }
}
