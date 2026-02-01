<?php

namespace App\Extensions\Chatbot\System\Console\Commands;

use App\Extensions\Chatbot\System\Models\Chatbot;
use App\Helpers\Classes\Helper;
use Illuminate\Console\Command;
use Illuminate\Support\Facades\Log;

class ClearDemoModeCommand extends Command
{
    protected $signature = 'app:clear-chatbot-demo-mode';

    protected $description = 'Clear chatbot demo mode';

    public function handle(): void
    {
        if (Helper::appIsNotDemo()) {
            return;
        }

        Log::info('Clear chatbot demo mode new');

        Chatbot::query()->where('is_demo', '=', 0)
            ->where('created_at', '<', now()->subMinutes(30))
            ->delete();
    }
}
