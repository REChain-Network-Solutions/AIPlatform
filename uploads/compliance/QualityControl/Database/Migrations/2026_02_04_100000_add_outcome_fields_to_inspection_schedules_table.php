<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        if (!Schema::hasTable('inspection_schedules')) {
            return;
        }

        Schema::table('inspection_schedules', function (Blueprint $table) {
            if (!Schema::hasColumn('inspection_schedules', 'qc_outcome')) {
                // pending | pass | fail | needs_reclean | escalated
                $table->string('qc_outcome')->nullable()->after('complaint_id');
            }
            if (!Schema::hasColumn('inspection_schedules', 'qc_outcome_set_at')) {
                $table->timestamp('qc_outcome_set_at')->nullable()->after('qc_outcome');
            }
            if (!Schema::hasColumn('inspection_schedules', 'follow_up_schedule_id')) {
                $table->unsignedInteger('follow_up_schedule_id')->nullable()->after('qc_outcome_set_at');
            }
        });
    }

    public function down(): void
    {
        if (!Schema::hasTable('inspection_schedules')) {
            return;
        }

        Schema::table('inspection_schedules', function (Blueprint $table) {
            foreach (['follow_up_schedule_id', 'qc_outcome_set_at', 'qc_outcome'] as $col) {
                if (Schema::hasColumn('inspection_schedules', $col)) {
                    $table->dropColumn($col);
                }
            }
        });
    }
};
