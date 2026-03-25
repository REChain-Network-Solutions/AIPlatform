<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        Schema::create('tasks', function (Blueprint $table) {
            $table->id();

            // Ownership
            $table->foreignId('user_id')->constrained()->cascadeOnDelete();

            // Core fields
            $table->string('title');
            $table->text('description')->nullable();
            $table->datetime('due_date')->nullable()->index();
            $table->datetime('completed_at')->nullable();
            $table->datetime('reminder_at')->nullable();

            // Status & Priority
            $table->foreignId('task_status_id')->nullable()->constrained('task_statuses')->nullOnDelete();
            $table->foreignId('task_priority_id')->nullable()->constrained('task_priorities')->nullOnDelete();

            // Assignment
            $table->foreignId('assigned_to_user_id')->nullable()->constrained('users')->nullOnDelete();
            $table->foreignId('completed_by_user_id')->nullable()->constrained('users')->nullOnDelete();

            // Polymorphic relation (linked to Contact, Company, Lead, Deal, Project, Job)
            $table->nullableMorphs('taskable');

            // Hierarchy (sub-tasks)
            $table->foreignId('parent_task_id')->nullable()->constrained('tasks')->nullOnDelete()->index();

            // Project management extras (from PMCore)
            $table->decimal('estimated_hours', 8, 2)->nullable();
            $table->decimal('actual_hours', 8, 2)->nullable();
            $table->unsignedInteger('task_order')->default(0);
            $table->boolean('is_milestone')->default(false);

            // AI metadata
            $table->json('ai_metadata')->nullable(); // stores AI-extracted entities, voice transcript, etc.

            $table->softDeletes();
            $table->timestamps();
        });
    }

    public function down(): void
    {
        Schema::dropIfExists('tasks');
    }
};
