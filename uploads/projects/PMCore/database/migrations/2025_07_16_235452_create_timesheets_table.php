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
        Schema::create('timesheets', function (Blueprint $table) {
            $table->id();
            $table->unsignedBigInteger('user_id');
            $table->unsignedBigInteger('project_id');
            $table->unsignedBigInteger('task_id')->nullable();
            $table->date('date');
            $table->decimal('hours', 4, 2);
            $table->text('description')->nullable();
            $table->boolean('is_billable')->default(true);
            $table->decimal('billing_rate', 8, 2)->nullable();
            $table->decimal('cost_rate', 8, 2)->nullable();
            $table->enum('status', ['draft', 'submitted', 'approved', 'rejected', 'invoiced'])->default('draft');
            $table->unsignedBigInteger('approved_by_id')->nullable();
            $table->timestamp('approved_at')->nullable();
            $table->unsignedBigInteger('created_by_id')->nullable();
            $table->unsignedBigInteger('updated_by_id')->nullable();
            $table->timestamps();
            $table->softDeletes();

            // Indexes
            $table->index(['user_id', 'date']);
            $table->index(['project_id', 'date']);
            $table->index(['status']);
            $table->index(['is_billable']);
            $table->index(['approved_by_id']);

            // Foreign key constraints
            $table->foreign('user_id')->references('id')->on('users');
            $table->foreign('project_id')->references('id')->on('projects');
            $table->foreign('task_id')->references('id')->on('crm_tasks');
            $table->foreign('approved_by_id')->references('id')->on('users');
            $table->foreign('created_by_id')->references('id')->on('users');
            $table->foreign('updated_by_id')->references('id')->on('users');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('timesheets');
    }
};
