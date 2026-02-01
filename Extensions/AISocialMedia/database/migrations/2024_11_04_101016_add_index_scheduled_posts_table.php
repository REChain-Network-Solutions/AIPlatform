<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        if (Schema::hasIndex('scheduled_posts', 'command_running_last_run_date_index')) {
            return;
        }

        Schema::table('scheduled_posts', static function (Blueprint $table) {
            $table->index(
                [
                    'command_running',
                    'last_run_date',
                    'repeat_period',
                    'repeat_time',
                ],
                'command_running_last_run_date_index'
            );
        });
    }

    public function down(): void {}
};
