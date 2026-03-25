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
        Schema::table('projects', function (Blueprint $table) {
            // Basic budget tracking fields
            $table->decimal('actual_cost', 15, 2)->default(0)->after('budget');
            $table->decimal('actual_revenue', 15, 2)->default(0)->after('actual_cost');

            // Progress tracking
            $table->integer('completion_percentage')->default(0)->after('actual_revenue');
            $table->timestamp('completed_at')->nullable()->after('completion_percentage');
            $table->boolean('is_archived')->default(false)->after('completed_at');

            // Add indexes for reporting
            $table->index(['is_archived', 'status']);
            $table->index('completed_at');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::table('projects', function (Blueprint $table) {
            $table->dropColumn([
                'actual_cost',
                'actual_revenue',
                'completion_percentage',
                'completed_at',
                'is_archived',
            ]);

            $table->dropIndex(['is_archived', 'status']);
            $table->dropIndex(['completed_at']);
        });
    }
};
