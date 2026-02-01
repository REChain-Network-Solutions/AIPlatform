<?php

namespace App\Console\Commands\Clear;

use App\Extensions\AIRealtimeImage\System\Models\RealtimeImage;
use App\Helpers\Classes\Helper;
use Illuminate\Console\Command;
use Illuminate\Support\Facades\Log;

class ClearAIRealtimeImageCommand extends Command
{
    private int $itemsToRetain = 22;

    protected $signature = 'app:clear-ai-realtime-image';

    protected $description = 'Command description';

    public function handle(): void
    {
        if (Helper::appIsNotDemo()) {
            return;
        }

        // Log::info('Clearing user ai real time data');

        RealtimeImage::query()
            ->where('is_demo', false)
            ->orderByDesc('id')
            ->where('created_at', '<', now()->subHour()) // Retain only the last 1 hours of data
            ->delete();

        Log::info('Clearing user ai real time data');
    }
}
