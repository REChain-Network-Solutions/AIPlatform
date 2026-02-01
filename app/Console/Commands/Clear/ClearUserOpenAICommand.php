<?php

namespace App\Console\Commands\Clear;

use App\Helpers\Classes\Helper;
use App\Models\UserOpenai;
use Illuminate\Console\Command;
use Illuminate\Support\Facades\Log;

class ClearUserOpenAICommand extends Command
{
    private int $itemsToRetain = 22;

    protected $signature = 'app:clear-user-open-a-i';

    protected $description = 'Command description';

    public function handle(): void
    {
        if (Helper::appIsNotDemo()) {
            return;
        }

        // Log::info('Clearing user OpenAI data (except last 22)...');

        $idsToRetain = UserOpenai::query()
            ->where('is_demo', false)
            ->orderByDesc('id')
            ->take($this->itemsToRetain)
            ->pluck('id')
            ->toArray();

        UserOpenai::query()
            ->where('is_demo', false)
            ->whereNotIn('id', $idsToRetain)
            ->where('openai_id', '!=', 36)
            ->orderByDesc('id')
            ->take(PHP_INT_MAX)
            ->delete();

        // Log::info('User OpenAI data cleared successfully (last 22 retained).');
    }
}
