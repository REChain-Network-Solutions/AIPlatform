<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        if (Schema::hasColumn('scheduled_posts', 'command_running')) {
            return;
        }

        Schema::table('scheduled_posts', static function (Blueprint $table) {
            $table->boolean('command_running')->default(false)->after('id');
            $table->date('last_run_date')
                ->useCurrent()
                ->nullable()
                ->after('command_running');
        });
    }

    public function down(): void {}
};
