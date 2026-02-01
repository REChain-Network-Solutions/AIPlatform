<?php

namespace App\Console\Commands\Clear;

use App\Helpers\Classes\Helper;
use Illuminate\Console\Command;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Log;

class ClearUserActivityCommand extends Command
{
    protected $signature = 'app:clear-user-activity';

    protected $description = 'Command description';

    public function handle(): void
    {
        if (Helper::appIsNotDemo()) {
            return;
        }

        // Log::info('Clearing user activity');

        // Assuming you have a UserActivity model and a user_activities table

        DB::table('users_activity')->truncate();

        DB::table('health_check_result_history_items')->truncate();

        // Add your logic to clear user activity here;

        // Log::info('User activity cleared successfully');
    }
}
