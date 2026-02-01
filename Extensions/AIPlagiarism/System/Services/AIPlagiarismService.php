<?php

namespace App\Extensions\AIPlagiarism\System\Services;

use App\Models\SettingTwo;
use Exception;
use GuzzleHttp\Client;
use GuzzleHttp\Exception\GuzzleException;
use Illuminate\Http\JsonResponse;

class AIPlagiarismService
{
    private const BASE_URL = 'https://plagiarismcheck.org/api/v1/';

    private const PLAGIARISM_ENDPOINT = 'text';

    private const PLAGIARISM_CHECK_ENDPOINT = 'text/report/';

    private const AI_DETECT_ENDPOINT = 'chat-gpt';

    private Client $client;

    private ?string $apiKey;

    public function __construct()
    {
        ini_set('max_execution_time', 3000);
        set_time_limit(3000);
        $this->apiKey = SettingTwo::getCache()->plagiarism_key ?? '';
        $this->client = new Client([
            'base_uri' => self::BASE_URL,
            'headers'  => [
                'Content-Type'  => 'application/x-www-form-urlencoded',
                'X-API-TOKEN'   => $this->apiKey,
            ],
        ]);
    }

    public function checkPlagiarism(string $text, string $lang = 'en'): JsonResponse
    {
        if (empty($this->apiKey)) {
            return response()->json(['message' => __('Please input plagiarism api key')], 401);
        }

        try {
            $data = [
                'language' => $lang,
                'text'     => $text,
            ];

            $response = $this->client->post(self::PLAGIARISM_ENDPOINT, [
                'form_params' => $data,
            ]);

            $result = json_decode($response->getBody()->getContents());
            if ($result->success === true) {
                $resultId = $result->data->text->id;
                while (true) {
                    $response = $this->client->get(self::PLAGIARISM_ENDPOINT . '/' . $resultId);
                    $result = json_decode($response->getBody()->getContents());

                    if (! empty($result->data->report)) {
                        break;
                    }
                }

                $response = $this->client->get(self::PLAGIARISM_CHECK_ENDPOINT . $resultId);

                return response()->json(json_decode($response->getBody()->getContents()));
            }

            return response()->json(['message', __('Error in plagiarism.org api')], 401);

        } catch (Exception|GuzzleException $e) {
            if ($e->hasResponse()) {
                $response = $e->getResponse();
                $errorMessage = $response->getBody()->getContents();

                return response()->json(['status' => 'error', 'message' => json_decode($errorMessage)?->message], 500);
            }

            return response()->json(['message' => $e->getMessage()], 500);
        }
    }

    public function detectAIContent(string $text, string $lang = 'en'): JsonResponse
    {
        if (empty($this->apiKey)) {
            return response()->json(['message' => __('Please input plagiarism api key')], 401);
        }

        try {
            $data = [
                'text'     => $text,
            ];

            $response = $this->client->post(self::AI_DETECT_ENDPOINT . '/', [
                'headers'     => [
                    'Content-Type'  => 'application/x-www-form-urlencoded',
                    'X-API-TOKEN'   => $this->apiKey,
                ],
                'form_params' => $data,
            ]);

            $result = json_decode($response->getBody()->getContents());
            if ($result->success === true) {
                $resultId = $result->data->id;
                while (true) {
                    $response = $this->client->get(self::AI_DETECT_ENDPOINT . '/' . $resultId);
                    $result = json_decode($response->getBody()->getContents());
                    if ($result->data->status === 4) {
                        break;
                    }
                }
                $response = $this->client->get(self::AI_DETECT_ENDPOINT . '/' . $resultId);

                return response()->json(json_decode($response->getBody()->getContents()));
            }

            return response()->json(['message', __('Error in plagiarism.org api')], 401);
        } catch (Exception|GuzzleException $e) {
            if ($e->hasResponse()) {
                $response = $e->getResponse();
                $errorMessage = $response->getBody()->getContents();

                return response()->json(['status' => 'error', 'message' => json_decode($errorMessage)?->message], 500);
            }

            return response()->json(['message' => $e->getMessage()], 500);
        }
    }
}
