<?php

namespace App\Extensions\AiPersona\System\Services;

use App\Extensions\AiPersona\System\Services\Traits\AiPersona;
use App\Helpers\Classes\Helper;
use App\Models\Setting;
use Exception;
use GuzzleHttp\Client;
use GuzzleHttp\Exception\RequestException;

class AiPersonaService
{
    use AiPersona;

    public const BASE_URL = 'https://api.heygen.com';

    public ?string $secretKey;

    private $client;

    public function __construct()
    {
        $this->secretKey = Setting::getCache()->heygen_secret_key;
        $this->client = new Client;
    }

    private function post($url, $params)
    {
        if (blank($this->secretKey)) {
            return [
                'error'  => 'API key is missing',
                'result' => [],
            ];
        }

        try {
            $response = $this->client->post(self::BASE_URL . $url, [
                'headers' => $this->getHeaders(),
                'body'    => json_encode($params),
            ])->getBody()->getContents();

            return json_decode($response, true);
        } catch (RequestException $e) {
            if ($e->hasResponse()) {
                $responseBody = $e->getResponse()->getBody()->getContents();
                $responseJson = json_decode($responseBody, true);

                return [
                    'error'   => $responseJson['error'] ?? 'Unknown error',
                    'message' => $responseJson['context'] ?? 'No context provided',
                ];
            }

            return [
                'error'   => 'Request failed without a response',
                'message' => 'No response context',
            ];
        } catch (Exception $e) {
            return [
                'error'   => 'An unexpected error occurred',
                'message' => $e->getMessage(),
            ];
        }
    }

    private function get($url)
    {
        if (blank($this->secretKey)) {
            return [];
        }

        try {
            $response = $this->client->get(Helper::parseUrl(self::BASE_URL, $url), [
                'headers' => $this->getHeaders(),
            ])->getBody()->getContents();

            return json_decode($response, true);
        } catch (Exception $e) {
            return [
                'error'  => 'API request failed: ' . $e->getMessage(),
                'videos' => [],
            ];
        }
    }

    private function delete($url)
    {
        if (blank($this->secretKey)) {
            return [];
        }

        return $this->client->delete(self::BASE_URL . $url, [
            'headers' => $this->getHeaders(),
        ]);
    }

    private function getHeaders(): array
    {
        return [
            'x-api-key'    => $this->secretKey,
            'accept'       => 'application/json',
            'content-type' => 'application/json',
        ];
    }
}
