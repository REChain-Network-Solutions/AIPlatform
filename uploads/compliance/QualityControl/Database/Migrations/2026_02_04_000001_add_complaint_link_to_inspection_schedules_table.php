<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        if (Schema::hasTable('inspection_schedules')) {
            Schema::table('inspection_schedules', function (Blueprint $table) {
                if (!Schema::hasColumn('inspection_schedules', 'complaint_id')) {
                    $table->unsignedInteger('complaint_id')->nullable()->after('job_id');
                }
            });
        }
    }

    public function down(): void
    {
        if (Schema::hasTable('inspection_schedules') && Schema::hasColumn('inspection_schedules', 'complaint_id')) {
            Schema::table('inspection_schedules', function (Blueprint $table) {
                $table->dropColumn('complaint_id');
            });
        }
    }
};
