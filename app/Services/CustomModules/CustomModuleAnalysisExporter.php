<?php
namespace App\Services\CustomModules;

use App\Services\CustomModules\DTO\ModuleAnalysisResult;
use Symfony\Component\HttpFoundation\StreamedResponse;

class CustomModuleAnalysisExporter
{
    public function download(ModuleAnalysisResult $analysis): StreamedResponse
    {
        return response()->streamDownload(function () use ($analysis) {
            echo json_encode($analysis->toArray(), JSON_PRETTY_PRINT);
        }, 'analysis.json', ['Content-Type' => 'application/json']);
    }
}
