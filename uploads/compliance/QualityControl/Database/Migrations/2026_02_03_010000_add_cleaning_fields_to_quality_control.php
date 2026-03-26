<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration {
    public function up(): void
    {
        // Link Quality Checks to cleaning jobs (loose coupling: no foreign key).
        if (Schema::hasTable('inspection_schedules')) {
            Schema::table('inspection_schedules', function (Blueprint $table) {
                if (!Schema::hasColumn('inspection_schedules', 'job_id')) {
                    $table->unsignedBigInteger('job_id')->nullable()->after('company_id');
                }
                if (!Schema::hasColumn('inspection_schedules', 'user_id')) {
                    $table->unsignedInteger('user_id')->nullable()->after('job_id');
                }
            });
        }

        // Proof photo categories (before/after/general).
        if (Schema::hasTable('inspection_schedule_files')) {
            Schema::table('inspection_schedule_files', function (Blueprint $table) {
                if (!Schema::hasColumn('inspection_schedule_files', 'category')) {
                    $table->string('category')->nullable()->default('general')->after('description');
                }
            });
        }
    }

    public function down(): void
    {
        if (Schema::hasTable('inspection_schedules')) {
            Schema::table('inspection_schedules', function (Blueprint $table) {
                if (Schema::hasColumn('inspection_schedules', 'job_id')) {
                    $table->dropColumn('job_id');
                }
                if (Schema::hasColumn('inspection_schedules', 'user_id')) {
                    $table->dropColumn('user_id');
                }
            });
        }

        if (Schema::hasTable('inspection_schedule_files')) {
            Schema::table('inspection_schedule_files', function (Blueprint $table) {
                if (Schema::hasColumn('inspection_schedule_files', 'category')) {
                    $table->dropColumn('category');
                }
            });
        }
    }
};
