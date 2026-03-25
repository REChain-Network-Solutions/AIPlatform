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
        Schema::create('resource_allocations', function (Blueprint $table) {
            $table->id();
            $table->unsignedBigInteger('user_id');
            $table->unsignedBigInteger('project_id');
            $table->date('start_date');
            $table->date('end_date')->nullable();
            $table->decimal('allocation_percentage', 5, 2)->default(100); // 0-100%
            $table->decimal('hours_per_day', 3, 1)->default(8.0); // Standard 8 hours
            $table->enum('allocation_type', ['project', 'task', 'phase'])->default('project');
            $table->unsignedBigInteger('task_id')->nullable();
            $table->string('phase')->nullable();
            $table->text('notes')->nullable();
            $table->boolean('is_billable')->default(true);
            $table->boolean('is_confirmed')->default(false);
            $table->enum('status', ['planned', 'active', 'completed', 'cancelled'])->default('planned');
            $table->unsignedBigInteger('created_by_id');
            $table->unsignedBigInteger('updated_by_id');
            $table->timestamps();
            $table->softDeletes();

            // Indexes
            $table->index('user_id');
            $table->index('project_id');
            $table->index(['start_date', 'end_date']);
            $table->index('status');
            $table->index(['user_id', 'start_date', 'end_date']);

            // Foreign keys (nullable for cross-module compatibility)
            $table->foreign('user_id')->references('id')->on('users')->onDelete('cascade');
            $table->foreign('project_id')->references('id')->on('projects')->onDelete('cascade');
        });

        // Create resource capacity table for tracking availability
        Schema::create('resource_capacities', function (Blueprint $table) {
            $table->id();
            $table->unsignedBigInteger('user_id');
            $table->date('date');
            $table->decimal('available_hours', 3, 1)->default(8.0);
            $table->decimal('allocated_hours', 3, 1)->default(0);
            $table->decimal('utilized_hours', 3, 1)->default(0); // From timesheets
            $table->boolean('is_working_day')->default(true);
            $table->string('leave_type')->nullable(); // holiday, vacation, sick, etc.
            $table->text('notes')->nullable();
            $table->timestamps();

            // Indexes
            $table->unique(['user_id', 'date']);
            $table->index('date');
            $table->index(['user_id', 'date']);

            // Foreign keys
            $table->foreign('user_id')->references('id')->on('users')->onDelete('cascade');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('resource_capacities');
        Schema::dropIfExists('resource_allocations');
    }
};
