<?php

namespace App\Http\Controllers;

use App\Services\Common\FontsService;
use GuzzleHttp\Exception\GuzzleException;
use Illuminate\Http\JsonResponse;
use JsonException;

class FontsController extends Controller
{
    /**
     * @throws GuzzleException
     * @throws JsonException
     */
    public function updateFontsCache(): JsonResponse
    {
        $dump = FontsService::updateFontsCache();

        return response()->json(['success' => true]);
    }
}
