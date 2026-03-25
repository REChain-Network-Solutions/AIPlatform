<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        Schema::table('timesheets', function (Blueprint $table) {
            // Add computed amount columns
            $table->decimal('cost_amount', 10, 2)->default(0)->after('cost_rate');
            $table->decimal('billable_amount', 10, 2)->default(0)->after('cost_amount');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::table('timesheets', function (Blueprint $table) {
            $table->dropColumn(['cost_amount', 'billable_amount']);
        });
    }
};
