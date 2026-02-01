<?php

namespace App\Extensions\AISocialMedia\System\Console\Commands\Concerns;

use App\Extensions\AISocialMedia\System\Jobs\UserPostJob;
use Exception;

trait HasDynamicHandle
{
    public function dynamicHandle(string $repeat_period): void
    {
        $posts = $this->query($repeat_period)->get();

        if ($posts->count()) {
            $this->query($repeat_period)->update(['command_running' => true, 'last_run_date'   => now()->toDateString()]);
        }

        foreach ($posts as $post) {
            try {
                dispatch(new UserPostJob($post));
            } catch (Exception $e) {
                continue;
            }
            $post->update(['command_running' => true]);
        }
    }
}
