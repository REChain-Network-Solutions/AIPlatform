<?php

namespace App\Extensions\SEOTool\System\Services\Search;

use App\Models\SettingTwo;
use App\Services\Contracts\BaseSearchService;
use GuzzleHttp\Client;
use Throwable;

class SerperDevSearch implements BaseSearchService
{
    public string $url = 'https://google.serper.dev/search';

    public ?string $apiKey = '';

    public function setApiKey(): void
    {
        $setting = SettingTwo::query()->first();

        $this->apiKey = $setting?->getAttribute('serper_api_key');
    }

    public function search($keyword)
    {
        $this->setApiKey();

        $client = new Client;
        $response = $client->post($this->url, [
            'headers' => [
                'X-API-KEY'    => $this->apiKey,
                'Content-Type' => 'application/json',
            ],
            'json'    => [
                'q' => $keyword,
            ],
        ]);

        $searchResult = $response->getBody()->getContents();

        try {
            $searchResult = json_decode($searchResult, false, 512, JSON_THROW_ON_ERROR);

        } catch (Throwable $th) {
        }

        return $searchResult;
    }

    public function getTopTitles($keyword): array
    {
        $searchResult = $this->search($keyword);

        try {
            $organics = $searchResult->organic;
        } catch (Throwable $th) {
            try {
                $organics = $searchResult['organic'];
            } catch (Throwable $th) {
                $organics = [];
            }
        }

        return collect($organics)->pluck('title')->toArray();
    }

    public function getKeywords($keyword): array
    {
        $searchResult = $this->search($keyword);

        try {
            $keywords = $searchResult->relatedSearches;
        } catch (Throwable $th) {
            try {
                $keywords = $searchResult['relatedSearches'];
            } catch (Throwable $th) {
                $keywords = [];
            }
        }

        return collect($keywords)->pluck('query')->toArray();
    }

    public function getTopStories($keyword)
    {
        $searchResult = $this->search($keyword);

        try {
            $topStories = $searchResult->topStories;
        } catch (Throwable $th) {

            if (isset($searchResult['topStories'])) {
                $topStories = $searchResult['topStories'];
            } else {
                $topStories = [];
            }
        }

        return collect($topStories)->pluck('title')->toArray();
    }

    public function getPeopleAlsoAsks($keyword): array
    {
        $searchResult = $this->search($keyword);

        try {
            $peopleAlsoAsk = $searchResult->peopleAlsoAsk;
        } catch (Throwable $th) {
            try {
                $peopleAlsoAsk = $searchResult['peopleAlsoAsk'];
            } catch (Throwable $th) {
                $peopleAlsoAsk = [];
            }
        }

        return collect($peopleAlsoAsk)->pluck('question')->toArray();
    }
}
