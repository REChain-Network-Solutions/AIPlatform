<?php

namespace App\Extensions\AISocialMedia\System\Console\Commands\Concerns;

use App\Extensions\AISocialMedia\System\Models\ScheduledPost;
use Illuminate\Database\Eloquent\Builder;
use Illuminate\Support\Facades\DB;

trait HasGenerateQuery
{
    public function query(string $repeat_period): Builder
    {
        $oneMinuteBefore = now()->subMinute()->format('H:i');

        $oneMinuteAfter = now()->addMinute()->format('H:i');

        return ScheduledPost::query()
            ->where('command_running', false)
            ->whereDate('last_run_date', '<', now()->format('Y-m-d'))
            ->where('repeat_period', '=', $repeat_period)
            ->whereBetween(
                DB::raw("DATE_FORMAT(`repeat_time`, '%H:%i')"),
                [$oneMinuteBefore, $oneMinuteAfter]
            );
    }
}
